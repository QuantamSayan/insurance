"""
Custom FunctionTransformer helpers for the insurance model pipeline.

IMPORTANT: This file must be kept IDENTICAL in both:
  - ml/transformers.py        (used when training/pickling the model)
  - api/models/transformers.py (used when loading the model in the API)

Any change here must be copied to the other location and the model
must be re-trained/re-pickled for the change to take effect.
"""


def smokers_fun(x):
    """
    Zeroes out the numeric feature for non-smokers, keeping it only
    for rows where the second column equals 'yes'.

    Expects a 2-column input: [numeric_feature, smoker_status]
    """
    feature_series = x.iloc[:, 0]
    smokers_series = x.iloc[:, 1]
    return (feature_series * ((smokers_series == "yes").astype(int))).to_frame()


def smokers_feature_name(function_transformer, feature_names_in):
    """Output feature name for smokers_fun, e.g. 'bmi_of_smokers'."""
    return [str(feature_names_in[0]) + "_of_smokers"]


def non_smokers_fun(x):
    """
    Zeroes out the numeric feature for smokers, keeping it only
    for rows where the second column equals 'no'.

    Expects a 2-column input: [numeric_feature, smoker_status]
    """
    feature_series = x.iloc[:, 0]
    non_smokers_series = x.iloc[:, 1]
    return (feature_series * ((non_smokers_series == "no").astype(int))).to_frame()


def non_smokers_feature_name(function_transformer, feature_names_in):
    """Output feature name for non_smokers_fun, e.g. 'bmi_of_non_smokers'."""
    return [str(feature_names_in[0]) + "_of_non_smokers"]