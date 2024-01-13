##############value results, event idx:32095_to_94390##############
AUC_fs = {
    'Wikipedia': {
        'TGAT': {
            'ATTN': 0.564(0.921),
            'PBONE': -2.227(4.212),
            'PG': 0.692(1.614),
            'ourexplainer': 1.477(0.991),
        },
        'TGN': {
            'ATTN': 0.073(0.966),
            'PBONE': -0.601(1.834),
            'PG': -0.231(1.432),
            'ourexplainer': 0.590(0.805),
        }
    },
    'Reddit': {
        'TGAT': {
            'ATTN': -0.654(3.639),
            'PBONE': -2.492(4.203),
            'PG': -0.369(2.810),
            'ourexplainer': 1.076(2.357),
        },
        'TGN': {
            'ATTN': 0.289(0.444),
            'PBONE': -0.256(1.217),
            'PG': 0.020(0.783),
            'ourexplainer': 1.113(0.833),
        }
    },
    'Simulate v1': {
        'TGAT': {
            'ATTN': 0.390(0.405),
            'PBONE': -2.882(4.283),
            'PG': -0.081(0.829),
            'ourexplainer': 0.666(0.499),
        },
        'TGN': {
            'ATTN': 0,
            'PBONE': 0,
            'PG': 0,
            'ourexplainer': 0,
        }
    },
    'Simulate v2': {
        'TGAT': {
            'ATTN': 0,
            'PBONE': 0,
            'PG': 0,
            'ourexplainer': 0,
        },
        'TGN': {
            'ATTN': 0,
            'PBONE': 0,
            'PG': 0,
            'ourexplainer': 0,
        }
    },
}


Best_fid = {
    'Wikipedia': {
        'TGAT': {
            'ATTN': 0.891(0.822),
            'PBONE': 0.027(0.013),
            'PG': 1.354(0.754),
            'ourexplainer': 1.836(0.937),
        },
        'TGN': {
            'ATTN': 0.479(0.409),
            'PBONE': 0.296(0.317),
            'PG': 0.464(0.325),
            'ourexplainer': 0.866(0.485),
        }
    },
    'Reddit': {
        'TGAT': {
            'ATTN': 0.658(1.606),
            'PBONE': 0.167(0.297),
            'PG': 0.804(1.154),
            'ourexplainer': 1.518(1.956),
        },
        'TGN': {
            'ATTN': 0.575(0.311),
            'PBONE': 0.340(0.247),
            'PG': 0.679(0.448),
            'ourexplainer': 1.362(0.846),
        }
    },
    'Simulate v1': {
        'TGAT': {
            'ATTN': 0.555(0.255),
            'PBONE': 0.044(0.044),
            'PG': 0.476(0.271),
            'ourexplainer': 0.780(0.494),
        },
        'TGN': {
            'ATTN': 0,
            'PBONE': 0,
            'PG': 0,
            'ourexplainer': 0,
        }
    },
    'Simulate v2': {
        'TGAT': {
            'ATTN': 0,
            'PBONE': 0,
            'PG': 0,
            'ourexplainer': 0,
        },
        'TGN': {
            'ATTN': 0,
            'PBONE': 0,
            'PG': 0,
            'ourexplainer': 0,
        }
    },
}