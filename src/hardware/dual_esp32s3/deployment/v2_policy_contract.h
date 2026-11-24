#ifndef CYBERDUCK_V2_POLICY_CONTRACT_H
#define CYBERDUCK_V2_POLICY_CONTRACT_H

// Include the exported quantization config and weights before this contract.
#if !defined(CYBERDUCK_V2_POLICY) || CYBERDUCK_V2_POLICY != 1
#error "MCU2 requires an exported CyberDuck V2 policy; run python duck.py export --target esp32 with a verified V2 checkpoint"
#endif
#if !defined(CYBERDUCK_V2_QUANT_CONFIG) || CYBERDUCK_V2_QUANT_CONFIG != 1
#error "MCU2 requires the CyberDuck V2 quantization config exported with its policy"
#endif
#if !defined(POLICY_MODEL_NAME) || !defined(QUANT_MODEL_NAME)
#error "CyberDuck V2 model identity is missing from the exported ESP32 headers"
#else
constexpr bool cyberduck_v2_model_name_matches(const char* actual, const char* expected) {
    return *actual == *expected && (*actual == '\0' || cyberduck_v2_model_name_matches(actual + 1, expected + 1));
}
static_assert(cyberduck_v2_model_name_matches(POLICY_MODEL_NAME, "cyberduck_v2.xml"), "MCU2 policy model identity must be cyberduck_v2.xml");
static_assert(cyberduck_v2_model_name_matches(QUANT_MODEL_NAME, "cyberduck_v2.xml"), "MCU2 quantization config model identity must be cyberduck_v2.xml");
static_assert(POLICY_LAYER1_IN == 37, "MCU2 policy contract requires 37 observations");
static_assert(OBS_DIM == 37, "MCU2 quantization config requires OBS_DIM=37");
static_assert(POLICY_OBS_SCHEMA_VERSION == 2, "MCU2 policy contract requires observation schema version 2");
#endif

#endif  // CYBERDUCK_V2_POLICY_CONTRACT_H
