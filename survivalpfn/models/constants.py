"""
Constants used throughout the SurvivalPFN codebase.

This module centralizes magic numbers and configuration constants to improve
maintainability and clarity.
"""

# Numerical stability constants
EPSILON_STABILITY = 1e-20  # Used for numerical stability in division/normalization

# Censoring rate constraints
MAX_CENSORING_RATE = 0.95  # Maximum censoring rate (too high rates prohibit inference)

# Statistical significance thresholds
SIGNIFICANCE_LEVEL = (
    0.05  # Standard p-value threshold for statistical tests (95% confidence)
)

# Choices for number of components (distributions) in mixture models
K_CHOICES = [3, 5, 10, 20, 50]

# Choices for types of distributions used in mixture models
DIST_CHOICES = ["Weibull", "LogNormal"]

# Choices for types of censoring dependency
# Uniform: drawn uniformly
# Tab: drawn from a 1-D table generator
# Admin: draw entry date from a 1-D table generator, administrative censoring at fixed date
# Cond Ind: conditional independence censoring
CENSORING_CHOICES = ["Uniform", "Tab", "Admin", "Cond Ind"]
