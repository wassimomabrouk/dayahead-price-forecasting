"""Configuration invariants. Cheap, but they catch registry edits."""

from dayahead import config as cfg


def test_roles_partition_the_registry():
    roles = {v["role"] for v in cfg.SERIES.values()}
    assert roles == {"target", "exog", "forbidden"}
    assert len(cfg.CORE) + len(cfg.FORBIDDEN) == len(cfg.SERIES)


def test_forbidden_series_are_not_in_core():
    assert not set(cfg.FORBIDDEN) & set(cfg.CORE)


def test_price_lag_respects_auction_structure():
    # Prices for delivery day D clear at the auction on D-1, so at issuance
    # the newest price is already a full day old in auction terms.
    assert cfg.MIN_PRICE_LAG_HOURS >= 24


def test_urls_are_well_formed():
    assert cfg.index_url("4169").endswith("/4169/DE/index_hour.json")
    assert cfg.block_url("4169", 123).endswith("/4169_DE_hour_123.json")
