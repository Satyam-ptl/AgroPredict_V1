#!/usr/bin/env bash
# Smoke-test the REAL running server -- your actual model + your actual OpenWeather key.
# Different from `pytest` (backend/tests/), which deliberately uses fake data so it
# doesn't depend on your model files or API key at all.
#
# Usage:
#   1. In one terminal: cd backend && uvicorn app.main:app --reload
#   2. In another terminal: bash smoke_test.sh
#
# Each check prints PASS/FAIL. A FAIL tells you exactly what's broken and why.

BASE_URL="${1:-http://127.0.0.1:8000}"
PASS=0
FAIL=0

check() {
  local name="$1"
  local condition="$2"
  if [ "$condition" = "0" ]; then
    echo "PASS: $name"
    PASS=$((PASS+1))
  else
    echo "FAIL: $name"
    FAIL=$((FAIL+1))
  fi
}

echo "Testing against $BASE_URL"
echo "---"

# 1. Is the server even up?
HEALTH=$(curl -s -o /tmp/health.json -w "%{http_code}" "$BASE_URL/health")
check "Server responds to /health" "$([ "$HEALTH" = "200" ] && echo 0 || echo 1)"
if [ "$HEALTH" = "200" ]; then
  echo "  -> $(cat /tmp/health.json)"
else
  echo "  -> Got HTTP $HEALTH. Is uvicorn running? Did registry.load() fail? Check the uvicorn terminal for a traceback."
  echo "  -> Common cause: model_final_pipeline.joblib / target_label_encoder.joblib / deploy_metadata.json"
  echo "     not present in backend/artifacts/ (or AGROPREDICT_ARTIFACT_DIR pointing elsewhere)."
fi

# 2. Did the real categorical options load correctly?
OPTIONS=$(curl -s -o /tmp/options.json -w "%{http_code}" "$BASE_URL/options")
check "/options returns real soil_type/season lists" "$([ "$OPTIONS" = "200" ] && echo 0 || echo 1)"
if [ "$OPTIONS" = "200" ]; then
  echo "  -> $(cat /tmp/options.json)"
fi

# 3. Real prediction with manual weather -- exercises the actual trained model
echo "  (enter one valid soil_type and season from the /options output above if this fails)"
PREDICT=$(curl -s -o /tmp/predict.json -w "%{http_code}" -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "nitrogen_N_kg_ha": 90,
    "phosphorus_P_kg_ha": 42,
    "potassium_K_kg_ha": 43,
    "temperature_C": 26.5,
    "humidity_percent": 80,
    "rainfall_mm": 220,
    "soil_pH": 6.5,
    "soil_moisture_percent": 45,
    "soil_type": "REPLACE_WITH_REAL_VALUE",
    "season": "REPLACE_WITH_REAL_VALUE"
  }')
check "/predict returns 200 with real model" "$([ "$PREDICT" = "200" ] && echo 0 || echo 1)"
if [ "$PREDICT" = "200" ]; then
  echo "  -> $(cat /tmp/predict.json | head -c 400)..."
else
  echo "  -> Got HTTP $PREDICT: $(cat /tmp/predict.json)"
  echo "  -> If this says 'value is not a valid enumeration member' or similar, you need to"
  echo "     edit this script's soil_type/season values to match what /options actually returned."
fi

# 4. Real OpenWeather key check
AUTO=$(curl -s -o /tmp/auto.json -w "%{http_code}" -X POST "$BASE_URL/predict/auto" \
  -H "Content-Type: application/json" \
  -d '{
    "nitrogen_N_kg_ha": 90,
    "phosphorus_P_kg_ha": 42,
    "potassium_K_kg_ha": 43,
    "soil_pH": 6.5,
    "soil_moisture_percent": 45,
    "soil_type": "REPLACE_WITH_REAL_VALUE",
    "season": "REPLACE_WITH_REAL_VALUE",
    "city": "Nagpur"
  }')
check "/predict/auto works with real OpenWeather key" "$([ "$AUTO" = "200" ] && echo 0 || echo 1)"
if [ "$AUTO" = "200" ]; then
  echo "  -> $(cat /tmp/auto.json | head -c 400)..."
else
  echo "  -> Got HTTP $AUTO: $(cat /tmp/auto.json)"
  if grep -q "OPENWEATHER_API_KEY" /tmp/auto.json 2>/dev/null; then
    echo "  -> Your .env isn't being loaded. Did you add 'from dotenv import load_dotenv; load_dotenv()'"
    echo "     to the top of app/main.py, and pip install python-dotenv?"
  elif [ "$AUTO" = "401" ]; then
    echo "  -> Key might not be activated yet -- new OpenWeather keys can take up to a few hours."
  fi
fi

echo "---"
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = "0" ] && exit 0 || exit 1
