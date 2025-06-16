# Connectome data dictionary specifying weighted, plastic, delayed connections between neuron populations

connectome = {
    "LA_E": {
        "BLA_E": {"weight": 0.8, "plasticity": 1.0, "delay_jitter": 5},
        "CeA_I": {"weight": -0.7, "plasticity": 1.0, "delay_jitter": 5},
        "vmPFC_E": {"weight": 0.4, "plasticity": 0.9, "delay_jitter": 3}
    },
    "BLA_E": {
        "CeA_I": {"weight": -0.9, "plasticity": 1.1, "delay_jitter": 4},
        "mPFC_E": {"weight": 0.6, "plasticity": 1.0, "delay_jitter": 4},
        "dACC_E": {"weight": 0.5, "plasticity": 0.95, "delay_jitter": 3}
    },
    "CeA_I": {
        "vmPFC_I": {"weight": -0.3, "plasticity": 1.0, "delay_jitter": 3},
        "mPFC_I": {"weight": -0.2, "plasticity": 1.0, "delay_jitter": 3}
    },
    "vmPFC_E": {
        "LA_E": {"weight": 0.5, "plasticity": 1.0, "delay_jitter": 4},
        "BLA_E": {"weight": 0.6, "plasticity": 1.0, "delay_jitter": 4},
        "CeA_I": {"weight": -0.7, "plasticity": 1.05, "delay_jitter": 5},
        "aBNST_E": {"weight": 0.5, "plasticity": 0.9, "delay_jitter": 3},
        "vmPFC_I": {"weight": 0.3, "plasticity": 1.0, "delay_jitter": 3}
    },
    "vmPFC_I": {
        "vmPFC_E": {"weight": -0.4, "plasticity": 1.0, "delay_jitter": 3},
        "vmPFC_I": {"weight": -0.2, "plasticity": 1.0, "delay_jitter": 3}
    },
    "mPFC_E": {
        "BLA_E": {"weight": 0.5, "plasticity": 1.0, "delay_jitter": 4},
        "CeA_I": {"weight": -0.6, "plasticity": 1.0, "delay_jitter": 4},
        "dACC_E": {"weight": 0.4, "plasticity": 1.0, "delay_jitter": 3}
    },
    "mPFC_I": {
        "mPFC_E": {"weight": -0.4, "plasticity": 1.0, "delay_jitter": 3},
        "mPFC_I": {"weight": -0.2, "plasticity": 1.0, "delay_jitter": 3}
    },
    "dACC_E": {
        "rACC_E": {"weight": 0.6, "plasticity": 1.0, "delay_jitter": 3},
        "mPFC_E": {"weight": 0.4, "plasticity": 1.0, "delay_jitter": 4},
        "vmPFC_E": {"weight": 0.5, "plasticity": 0.95, "delay_jitter": 3}
    },
    "dACC_I": {
        "dACC_E": {"weight": -0.4, "plasticity": 1.0, "delay_jitter": 3}
    },
    "rACC_E": {
        "DLPFC_E": {"weight": 0.5, "plasticity": 1.0, "delay_jitter": 3}
    },
    "DLPFC_E": {
        "mPFC_E": {"weight": 0.6, "plasticity": 1.0, "delay_jitter": 4},
        "OFC_E": {"weight": 0.7, "plasticity": 1.0, "delay_jitter": 4},
        "dACC_E": {"weight": 0.6, "plasticity": 0.95, "delay_jitter": 3}
    },
    "OFC_E": {
        "vmPFC_E": {"weight": 0.7, "plasticity": 1.0, "delay_jitter": 3},
        "BLA_E": {"weight": 0.4, "plasticity": 1.0, "delay_jitter": 4}
    },
    "DG_E": {
        "CA3_E": {"weight": 0.9, "plasticity": 1.1, "delay_jitter": 3}
    },
    "CA3_E": {
        "CA1_E": {"weight": 0.8, "plasticity": 1.0, "delay_jitter": 3}
    },
    "CA1_E": {
        "Subiculum_E": {"weight": 0.9, "plasticity": 1.0, "delay_jitter": 3},
        "mPFC_E": {"weight": 0.4, "plasticity": 1.0, "delay_jitter": 4}
    },
    "Subiculum_E": {
        "vmPFC_E": {"weight": 0.5, "plasticity": 0.9, "delay_jitter": 3},
        "BLA_E": {"weight": 0.3, "plasticity": 1.0, "delay_jitter": 4}
    },
    "aBNST_E": {
        "CeA_I": {"weight": -0.4, "plasticity": 1.0, "delay_jitter": 3},
        "vmPFC_E": {"weight": 0.5, "plasticity": 0.95, "delay_jitter": 3}
    },
    "pBNST_E": {
        "CeA_I": {"weight": -0.3, "plasticity": 1.0, "delay_jitter": 3}
    },
    "AI_E": {
        "PI_E": {"weight": 0.6, "plasticity": 1.0, "delay_jitter": 3},
        "vmPFC_E": {"weight": 0.4, "plasticity": 1.0, "delay_jitter": 4}
    },
    "PI_E": {
        "vmPFC_E": {"weight": 0.3, "plasticity": 1.0, "delay_jitter": 3}
    },
    "dPAG_E": {
        "vlPAG_E": {"weight": 0.7, "plasticity": 1.0, "delay_jitter": 3},
        "lPAG_E": {"weight": 0.6, "plasticity": 1.0, "delay_jitter": 3}
    },
    "vlPAG_E": {
        "CeA_I": {"weight": -0.5, "plasticity": 1.0, "delay_jitter": 3}
    },
    "lPAG_E": {
        "CeA_I": {"weight": -0.5, "plasticity": 1.0, "delay_jitter": 3}
    }
}
