import polars as pl
import numpy as np
import fastlibrary as fl

from color_map import false_color_map_with_histogram


# read data function to read the MiQaT_Specification_Data file
file = R"T:\asm\EXE_ZdMB\10_EXE_MB_Flatness_Database\Flatness Reports\MiQaT_Specification_Data.parquet"
measurement_information = pl.read_parquet(file)

# select only measurements from sides ZA and ZE and certain process steps
measurements_ZAZE = (
    measurement_information.filter(
        (pl.col("Side").str.contains("ZA|ZE"))
        & (pl.col("Identifier") == "GridPV")
        & (pl.col("ProcessStep").str.contains("after IBF|after coating|after HI bonding|after Z polishing|before IBF"))
    )
    # remove "after IBF measurement which are not taken after the last IBF run"
    .filter(
        ~(
            pl.col("ProcessStep").str.contains("after IBF run").fill_null(False)
            & ~pl.col("Tags").str.contains("after IBF").fill_null(False)
        )
    )
    # select only the columns we need
    .select(["Serial", "ProcessStep", "Tags", "Side", "MeasurementDate", "SourcePath"])
    .unique()
).sort(by="MeasurementDate")

# remove all duplicate measurements - keep only the latest measurement for the same combination of serial, process step and side
measurements_ZAZE = measurements_ZAZE.group_by(["Serial", "ProcessStep", "Side"]).last()

print(measurements_ZAZE)