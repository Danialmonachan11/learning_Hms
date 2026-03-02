import polars as pl
import numpy as np
import fastlibrary as fl

def read_data(file_name: str) -> fl.Topography:
    topo = fl.read_zygo(file_name)
    return topo
from color_map import false_color_map_with_histogram

test_file = R"T:\asm\E X E_MB\1054117_EXE_MB_Zerodur_IBF_Coating\EX1-B30105\030_1054117_IFM_nISA02\04_Ebenheit_Final\Diff(VZK_yFlip_KA(EX1-B30105_$Seite_ZA$B$_02_nISA)_Gravkorr_270.wrk"
test_file=R"T:\asm\EXE_ZdMB\10_EXE_MB_Flatness_Database\EX1-B3028\2024-10-03 - 09h29 - after HI bonding #4\01 - Side data\EX1-B3028_$Seite_ZA$B$_M1_nHI.wrk"

test_data = fl.read_zygo(test_file)
test_data

false_color_map_with_histogram(test_data, title="Test data", z_min=-50, z_max=50)

file = R"T:\asm\EXE_ZdMB\10_EXE_MB_Flatness_Database\Flatness Reports\MiQaT_Specification_Data.parquet"
measurement_information = pl.read_parquet(file)
measurement_information
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

measurements_ZAZE
measurements_ZAZE = measurements_ZAZE.group_by(["Serial", "ProcessStep", "Side"]).last()
measurements_ZAZE
test_file1 = measurements_ZAZE.filter(
    (pl.col("Serial") == "EX1-B30105") & (pl.col("Side") == "ZA") & (pl.col("Tags").str.contains("after IBF"))
)["SourcePath"].item(0)
print(test_file1)

test_file2 = measurements_ZAZE.filter(
    (pl.col("Serial") == "EX1-B30105") & (pl.col("Side") == "ZA") & (pl.col("ProcessStep") == "after coating")
)["SourcePath"].item(0)
test_file2

topo1 = fl.read_zygo(test_file1)
display(topo1)
topo2 = fl.read_zygo(test_file2)
display(topo2)

fl.false_color_map(topo1, title="test measurement 1").show()
fl.false_color_map(topo2, title="test measurement 2").show()