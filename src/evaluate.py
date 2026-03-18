from __future__ import annotations

import csv
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils.status import normalize_status_label

GT_CSV = BASE_DIR / "data" / "ground_truth.csv"
RESULT_CSV = BASE_DIR / "data" / "results.csv"
FAILURE_CSV = BASE_DIR / "data" / "failure_cases.csv"
PLOT_DIR = BASE_DIR / "data" / "plots"
LOT_SEPARATOR = "|"


def get_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ModuleNotFoundError:
        return None


def load_csv_as_dict(csv_path: Path, key_field: str) -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            data[row[key_field]] = row
    return data


def char_accuracy(true_text: str, pred_text: str) -> float:
    true_text = true_text.strip()
    pred_text = pred_text.strip()

    if not true_text:
        return 1.0 if not pred_text else 0.0

    match_count = 0
    for index, char in enumerate(true_text):
        if index < len(pred_text) and pred_text[index] == char:
            match_count += 1

    return match_count / len(true_text)


def split_lots(lot_text: str) -> list[str]:
    return [token.strip() for token in lot_text.split(LOT_SEPARATOR) if token.strip()]


def normalize_lots_text(lot_text: str) -> str:
    return LOT_SEPARATOR.join(sorted(split_lots(lot_text)))


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def write_failure_cases(failure_cases: list[dict[str, str]]) -> None:
    fieldnames = [
        "image_name",
        "true_count",
        "pred_count",
        "true_status",
        "pred_status",
        "true_lots",
        "pred_lots",
        "count_match",
        "status_match",
        "lot_match",
    ]
    with FAILURE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failure_cases)


def save_metric_bar_chart(metric_values: dict[str, float]) -> Path:
    plt = get_pyplot()
    if plt is None:
        return Path("matplotlib not installed")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / "summary_metrics.png"

    labels = list(metric_values.keys())
    values = list(metric_values.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=["#355C7D", "#6C5B7B", "#C06C84", "#F67280", "#F8B195"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Evaluation Summary Metrics")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_status_accuracy_chart(status_accuracy_by_class: dict[str, float]) -> Path:
    plt = get_pyplot()
    if plt is None:
        return Path("matplotlib not installed")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / "status_accuracy.png"

    labels = list(status_accuracy_by_class.keys())
    values = list(status_accuracy_by_class.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color="#4C956C")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Status Class")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def save_confusion_chart(tp: int, fp: int, fn: int, tn: int) -> Path:
    plt = get_pyplot()
    if plt is None:
        return Path("matplotlib not installed")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / "abnormal_confusion_counts.png"

    labels = ["TP", "FP", "FN", "TN"]
    values = [tp, fp, fn, tn]
    colors = ["#2A9D8F", "#E76F51", "#F4A261", "#457B9D"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel("Image Count")
    plt.title("Abnormal Detection Confusion Counts")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.05, str(value), ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return output_path


def main() -> None:
    gt_data = load_csv_as_dict(GT_CSV, "image_name")
    pred_data = load_csv_as_dict(RESULT_CSV, "image_name")

    total = 0
    count_exact_match = 0
    count_abs_error_sum = 0
    status_correct = 0
    ocr_exact_match = 0
    ocr_char_acc_sum = 0.0
    end_to_end_success = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    status_totals: dict[str, int] = {}
    status_correct_counts: dict[str, int] = {}
    failure_cases: list[dict[str, str]] = []

    for image_name, gt in gt_data.items():
        if image_name not in pred_data:
            print(f"[WARNING] Missing prediction for: {image_name}")
            continue

        pred = pred_data[image_name]
        total += 1

        true_count = int(gt["true_count"])
        pred_count = int(pred["pred_count"])
        true_status = normalize_status_label(gt["true_status"])
        pred_status = normalize_status_label(pred["pred_status"])
        true_lots = normalize_lots_text(gt["true_lots"])
        pred_lots = normalize_lots_text(pred.get("pred_lots", pred.get("pred_lot", "")))
        status_totals[true_status] = status_totals.get(true_status, 0) + 1

        count_match = true_count == pred_count
        status_match = true_status == pred_status
        lot_match = true_lots == pred_lots

        if count_match:
            count_exact_match += 1
        count_abs_error_sum += abs(true_count - pred_count)

        if status_match:
            status_correct += 1
            status_correct_counts[true_status] = status_correct_counts.get(true_status, 0) + 1

        true_is_abnormal = true_status != "normal"
        pred_is_abnormal = pred_status != "normal"
        if true_is_abnormal and pred_is_abnormal:
            tp += 1
        elif not true_is_abnormal and pred_is_abnormal:
            fp += 1
        elif true_is_abnormal and not pred_is_abnormal:
            fn += 1
        else:
            tn += 1

        if lot_match:
            ocr_exact_match += 1
        ocr_char_acc_sum += char_accuracy(true_lots, pred_lots)

        if count_match and status_match and lot_match:
            end_to_end_success += 1
        else:
            failure_cases.append(
                {
                    "image_name": image_name,
                    "true_count": str(true_count),
                    "pred_count": str(pred_count),
                    "true_status": true_status,
                    "pred_status": pred_status,
                    "true_lots": true_lots,
                    "pred_lots": pred_lots,
                    "count_match": str(count_match),
                    "status_match": str(status_match),
                    "lot_match": str(lot_match),
                }
            )

    if total == 0:
        print("No evaluable rows found.")
        return

    count_exact_match_acc = safe_div(count_exact_match, total)
    count_mae = count_abs_error_sum / total
    status_acc = safe_div(status_correct, total)
    abnormal_precision = safe_div(tp, tp + fp)
    abnormal_recall = safe_div(tp, tp + fn)
    abnormal_f1 = safe_div(2 * abnormal_precision * abnormal_recall, abnormal_precision + abnormal_recall)
    ocr_exact_match_acc = safe_div(ocr_exact_match, total)
    ocr_char_acc_avg = ocr_char_acc_sum / total
    e2e_success_rate = safe_div(end_to_end_success, total)
    status_accuracy_by_class = {
        status_name: safe_div(status_correct_counts.get(status_name, 0), status_total)
        for status_name, status_total in sorted(status_totals.items())
    }
    write_failure_cases(failure_cases)
    metric_plot_path = save_metric_bar_chart(
        {
            "Count Exact": count_exact_match_acc,
            "Status Acc": status_acc,
            "OCR Exact": ocr_exact_match_acc,
            "OCR Char": ocr_char_acc_avg,
            "End-to-End": e2e_success_rate,
        }
    )
    status_plot_path = save_status_accuracy_chart(status_accuracy_by_class)
    confusion_plot_path = save_confusion_chart(tp, fp, fn, tn)

    print("\n========== Evaluation ==========")
    print(f"Total images                : {total}")
    print()
    print("[Count]")
    print(f"Count exact match accuracy  : {count_exact_match_acc:.4f}")
    print(f"Count MAE                   : {count_mae:.4f}")
    print()
    print("[Status]")
    print(f"Status accuracy             : {status_acc:.4f}")
    for status_name, status_accuracy in status_accuracy_by_class.items():
        print(f"Status accuracy ({status_name:<8}) : {status_accuracy:.4f}")
    print()
    print("[Abnormal Detection]")
    print(f"TP                          : {tp}")
    print(f"FP                          : {fp}")
    print(f"FN                          : {fn}")
    print(f"TN                          : {tn}")
    print(f"Abnormal precision          : {abnormal_precision:.4f}")
    print(f"Abnormal recall             : {abnormal_recall:.4f}")
    print(f"Abnormal F1-score           : {abnormal_f1:.4f}")
    print()
    print("[OCR]")
    print(f"OCR exact match accuracy    : {ocr_exact_match_acc:.4f}")
    print(f"OCR character accuracy      : {ocr_char_acc_avg:.4f}")
    print()
    print("[End-to-End]")
    print(f"End-to-end success rate     : {e2e_success_rate:.4f}")
    print(f"Failure case count          : {len(failure_cases)}")
    print(f"Failure case csv            : {FAILURE_CSV}")
    print(f"Metric plot                 : {metric_plot_path}")
    print(f"Status plot                 : {status_plot_path}")
    print(f"Confusion plot              : {confusion_plot_path}")
    if failure_cases:
        print("Failure image list          :")
        for failure in failure_cases:
            print(
                f"  - {failure['image_name']} "
                f"(count: {failure['true_count']}->{failure['pred_count']}, "
                f"status: {failure['true_status']}->{failure['pred_status']}, "
                f"lots: {failure['true_lots']}->{failure['pred_lots']})"
            )
    print("================================")


if __name__ == "__main__":
    main()
