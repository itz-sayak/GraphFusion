from backend.core.models import SemanticType
from backend.core.values import detect_date_formats, is_dayfirst, parse_currency, to_base_currency
from backend.ingestion.loader import ingest_file
from backend.profiling.profiler import profile_dataset
from backend.profiling.sketches import overlap_stats


def _profile(workspace, config, path):
    art = ingest_file(workspace, path)
    return profile_dataset(art, config)


def test_statistics_and_semantic_types(workspace, config, sample_files):
    prof, sketches = _profile(workspace, config, sample_files[0])
    cols = {c.name: c for c in prof.columns}
    assert prof.duplicate_rows == 12
    assert cols["customer_id"].semantic_type == SemanticType.ID
    assert cols["email"].semantic_type == SemanticType.EMAIL
    assert cols["city"].semantic_type == SemanticType.CITY
    assert cols["customer_name"].semantic_type == SemanticType.NAME
    assert cols["signup_date"].semantic_type == SemanticType.DATE and cols["signup_date"].date_dayfirst
    assert cols["phone"].null_pct > 0
    assert cols["customer_id"].min is not None and cols["customer_id"].mean is not None
    assert sketches["city"].complete and "mumbai" in sketches["city"].values


def test_currency_and_datetime_types(workspace, config, sample_files):
    sales, _ = _profile(workspace, config, sample_files[2])
    master, _ = _profile(workspace, config, sample_files[1])
    assert sales.column("amount_spent").semantic_type == SemanticType.CURRENCY
    assert master.column("last_updated").semantic_type == SemanticType.DATETIME


def test_value_helpers():
    assert parse_currency("₹1,200.50") == (1200.5, "INR")
    assert parse_currency("45.00 EUR") == (45.0, "EUR")
    assert parse_currency("USD 12") == (12.0, "USD")
    assert to_base_currency(100, "INR") == 1.17
    rate, fmt, has_time = detect_date_formats(["2024-01-05", "23/03/2024", "12 Jan 2024"])
    assert rate == 1.0 and not has_time
    assert is_dayfirst(["04/05/2024", "23/03/2024"]) and not is_dayfirst(["04/05/2024", "12/31/2024"])


def test_overlap_is_exact_for_complete_sketches(workspace, config, tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("city\n" + "\n".join(["Mumbai", "Delhi", "Pune", "Chennai"]) + "\n")
    b.write_text("location\n" + "\n".join(["mumbai", "DELHI", "Bengaluru"]) + "\n")
    _, sa = _profile(workspace, config, a)
    _, sb = _profile(workspace, config, b)
    st = overlap_stats(sa["city"], sb["location"])
    assert st["intersection"] == 2
    assert abs(st["containment_right"] - 2 / 3) < 1e-9
