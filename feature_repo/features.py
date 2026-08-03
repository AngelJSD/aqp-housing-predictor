"""Entity + FeatureView for `arequipa_listings_features`.

Matches ml/feature_metadata.csv exactly (task 2's catalog) — same 5
features, same feast_feature_view name, nothing added or removed. Source
table is `features` in Postgres (see ml/data_prep/load_features_to_postgres.py).
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float64, String
from feast.value_type import ValueType

# id is a hashed string (e.g. "4oBAQqoLR7EEEN85uot6+g=="), not numeric.
listing = Entity(name="listing", join_keys=["id"], value_type=ValueType.STRING)

features_source = PostgreSQLSource(
    name="features_source",
    table="features",
    timestamp_field="created_on",
)

# ttl=0 (forever): each row is one listing's own point-in-time snapshot
# keyed by `id`, not a value that decays with staleness like a rolling
# aggregate would.
arequipa_listings_features = FeatureView(
    name="arequipa_listings_features",
    entities=[listing],
    ttl=timedelta(days=0),
    schema=[
        Field(name="district", dtype=String),
        Field(name="surface", dtype=Float64),
        Field(name="property_type", dtype=String),
        Field(name="operation_type", dtype=String),
        Field(name="district_avg_price_per_m2", dtype=Float64),
    ],
    source=features_source,
    online=True,
)
