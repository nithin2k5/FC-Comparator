from fc_comparator.barcode import parse_scan
from fc_comparator.config import BarcodeConfig

KNOWN = {"P001", "P002"}


def test_part_scan():
    cfg = BarcodeConfig()
    assert parse_scan("P001\r\n", cfg, KNOWN).kind == "part"
    assert parse_scan("p002", cfg, KNOWN).value == "P002"  # case-insensitive match to a known part
    assert parse_scan("P999", cfg, KNOWN).kind == "unknown"
    assert parse_scan("P999", cfg).kind == "part"  # no known list: accept anything matching


def test_operator_scan():
    cfg = BarcodeConfig()
    s = parse_scan("OP:4711", cfg, KNOWN)
    assert (s.kind, s.value) == ("operator", "4711")
    assert parse_scan("op: 12 ", cfg, KNOWN).value == "12"
    assert parse_scan("OP:", cfg, KNOWN).kind == "unknown"


def test_custom_pattern_extracts_part_from_label():
    cfg = BarcodeConfig(part_pattern=r"PN=(?P<part>P\d{3});")
    assert parse_scan("LOT=88;PN=P002;QTY=4", cfg, KNOWN).value == "P002"
    assert parse_scan("garbage", cfg, KNOWN).kind == "unknown"
