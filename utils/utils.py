import pandas as pd
import numpy as np


def fill_NaN_and_Infs(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy[column] = df_copy[column].apply(
        lambda x: 0 if np.isnan(x) or np.isinf(x) else x
    )
    return df_copy


def fill_NaN_and_Infs_all(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for column in df_copy.columns:
        df_copy = fill_NaN_and_Infs(df_copy, column)
    return df_copy


def replace_nao_aplicavel(df: pd.DataFrame, columns: list[str]
                          ) -> pd.DataFrame:
    df_copy = df.copy()
    for column in columns:
        df_copy[column] = df_copy[column].replace("Não aplicável", 0)
        df_copy[column] = df_copy[column].astype(int)
    return df_copy


def get_mean_columns(
    df: pd.DataFrame, first_column: str, second_column: str
) -> pd.Series:
    df_copy = df[[first_column, second_column]].copy()
    df_copy["MDA"] = df_copy[first_column] / df_copy[second_column]
    df_copy = fill_NaN_and_Infs(df_copy, "MDA")
    return df_copy["MDA"]


def get_mean_columns_pivoted(
    df: pd.DataFrame,
    mean_columns: list[str],
    pivot_column: str,
) -> pd.DataFrame:
    df_copy = df[["NO_MUNICIPIO", "ANO", pivot_column] + mean_columns].copy()
    df_copy = replace_nao_aplicavel(df_copy, mean_columns)
    df_copy = df_copy.groupby(
        by=["NO_MUNICIPIO", "ANO", pivot_column], as_index=False
    ).sum(numeric_only=True)
    df_copy["MDA"] = get_mean_columns(df_copy, mean_columns[0],
                                      mean_columns[1])
    df_copy = df_copy.pivot_table(
        index=["NO_MUNICIPIO", "ANO"],
        columns=pivot_column,
        values="MDA",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    return df_copy


def replace_sim_nao_to_binary(df: pd.DataFrame, columns: list[str]
                              ) -> pd.DataFrame:
    df_copy = df.copy()
    for column in columns:
        df_copy[column] = df_copy[column].replace(r"\bSim\b", 1, regex=True)
        df_copy[column] = df_copy[column].replace(r"\bNão\b", 0, regex=True)
        df_copy[column] = df_copy[column].astype(int)
    return df_copy


def get_prop_columns(
    df: pd.DataFrame, first_column: str, second_column: str
) -> pd.Series:
    df_copy = df[[first_column, second_column]].copy()
    df_copy["PROP"] = df_copy[first_column] / df_copy[second_column]
    df_copy = fill_NaN_and_Infs(df_copy, "PROP")
    df_copy["PROP"] = df_copy["PROP"].apply(lambda x: 0 if x > 1 else x)
    df_copy["PROP"] = df_copy["PROP"].round(2)
    return df_copy["PROP"]


def get_qt_estabelecimento_columns_pivoted(
    df: pd.DataFrame,
    qt_column: str,
    pivot_column: str,
) -> pd.DataFrame:
    df_copy = df[["NO_MUNICIPIO", "ANO", pivot_column] + [qt_column]].copy()
    df_copy = replace_nao_aplicavel(df_copy, [qt_column])
    df_copy["QT"] = df_copy[qt_column].apply(
        lambda x: 1 if x > 0 else 0
    )
    df_copy = df_copy.groupby(by=["NO_MUNICIPIO", "ANO", pivot_column],
                              as_index=False).sum(
        numeric_only=True
    ).drop(columns=qt_column)
    df_copy = df_copy.pivot_table(index=["NO_MUNICIPIO", "ANO"],
                                  columns=pivot_column, values='QT',
                                  aggfunc='sum', fill_value=0).reset_index()

    return df_copy
