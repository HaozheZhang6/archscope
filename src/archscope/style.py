"""Visual design system: palette, typography, edge styles.

Color semantics follow the conventions of hand-made NeurIPS/ICLR figures:
one pastel family per component kind, darker stroke, near-black text.
"""

FONT_SANS = "Helvetica Neue, Helvetica, Arial, sans-serif"
FONT_MONO = "SF Mono, Menlo, Consolas, monospace"

INK = "#0F172A"
MUTED = "#475569"
FAINT = "#94A3B8"
BG = "#FFFFFF"

# kind -> (fill, stroke, text)
KIND = {
    "attention": ("#DBEAFE", "#2563EB", "#1E3A8A"),
    "ffn":       ("#DCFCE7", "#16A34A", "#14532D"),
    "norm":      ("#FEF9C3", "#CA8A04", "#713F12"),
    "linear":    ("#EDE9FE", "#7C3AED", "#4C1D95"),
    "vae":       ("#FFEDD5", "#EA580C", "#7C2D12"),
    "cond":      ("#FCE7F3", "#DB2777", "#831843"),
    "io":        ("#F8FAFC", "#64748B", "#334155"),
    "op":        ("#FFFFFF", "#475569", "#334155"),
    "mask":      ("#CCFBF1", "#0D9488", "#134E4A"),
    "head":      ("#E2E8F0", "#475569", "#1E293B"),
    "model":     ("#F1F5F9", "#475569", "#0F172A"),
    "known":     ("#F0FDF4", "#16A34A", "#166534"),
}

KIND_NAMES = {
    "attention": "Attention",
    "ffn": "MLP / FFN",
    "norm": "Normalization",
    "linear": "Linear / Embedding",
    "vae": "VAE / Conv",
    "cond": "Conditioning",
    "io": "Tensor / IO",
    "mask": "Masking / Routing",
    "head": "Output head",
    "known": "Known concept (collapsed)",
}

# modality -> (fill, stroke, text); used for token cells, lane tints, accents
MODALITY = {
    "video":  ("#E0F2FE", "#0284C7", "#0C4A6E"),
    "action": ("#D1FAE5", "#059669", "#064E3B"),
    "text":   ("#FCE7F3", "#DB2777", "#831843"),
    "state":  ("#FEF3C7", "#D97706", "#78350F"),
    "none":   ("#F1F5F9", "#94A3B8", "#334155"),
}

EDGE = {
    "main":     dict(stroke="#334155", width=1.5, dash=None),
    "residual": dict(stroke="#94A3B8", width=1.3, dash=None),
    "cond":     dict(stroke="#DB2777", width=1.2, dash="5 3"),
    "cache":    dict(stroke="#0D9488", width=1.3, dash="5 3"),
    "faint":    dict(stroke="#CBD5E1", width=1.1, dash=None),
}

EDGE_NAMES = {
    "main": "data flow",
    "residual": "residual",
    "cond": "conditioning",
    "cache": "cache / loop",
    "faint": "aux / discarded",
}

MODALITY_NAMES = {
    "video": "video", "action": "action", "text": "text",
    "state": "state", "none": "other",
}

# typography scale (px)
T_TITLE = 17
T_SUBTITLE = 11.5
T_SECTION = 13
T_LABEL = 12.5
T_SUB = 9.5
T_EDGE = 9.5
T_NOTE = 10.5
T_TINY = 8.5
