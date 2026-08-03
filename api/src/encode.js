// Mirrors pandas' `.cat.codes`: a category's position in the training-time
// list is its code; a value not in that list (never seen in train) becomes
// NaN — same "missing" value XGBoost/ONNX already routes natively, so an
// unseen district doesn't need special-casing beyond this.
function categoryCode(value, categories) {
  const idx = categories.indexOf(value);
  return idx === -1 ? NaN : idx;
}

// FEATURE_COLS order from ml/training/train.py: district, surface, property_type.
export function encodeInput({ district, surface, property_type }, categories) {
  return Float32Array.from([
    categoryCode(district, categories.district),
    surface,
    categoryCode(property_type, categories.property_type),
  ]);
}
