# ruff: noqa
"""Summarise the kernel performance from ncu profiles."""

import pathlib
import sys

import click
import pandas as pd


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
@click.command()
@click.argument(
    "folder",
    type=click.Path(
        file_okay=False, dir_okay=True, exists=True, path_type=pathlib.Path
    ),
    default="ncu_reports",
)
@click.option(
    "--repeats",
    default=1,
    type=int,
    show_default=True,
    help="How many times each kernel was replayed (ncu does this internally).",
)
def main(folder: pathlib.Path, repeats: int) -> None:
    """NCU profile summariser.

    Summarise dram__bytes_read.sum and dram__bytes_write.sum for every
    *.csv file inside FOLDER and write the result to summary.csv.
    """
    csv_files = sorted(folder.glob("**/*.csv"))
    if not csv_files:
        click.echo("No .csv files found in the given folder.", err=True)
        sys.exit(1)

    records = []
    for f in csv_files:
        # Remove summary.csv
        if f.name == "summary.csv":
            continue

        try:
            # Read the CSV with pandas directly (no skiprows)
            raw_df = pd.read_csv(f)

            # Debug output
            click.echo(f"Processing file: {f}")
            click.echo(f"Raw DataFrame shape: {raw_df.shape}")

            # Check for units in the second row
            read_unit = None
            write_unit = None
            data_df = raw_df  # Default if no unit row found

            # Check if second row contains units
            if len(raw_df) > 1:
                # Get second row values for read/write columns
                first_row_read = raw_df.iloc[0]["dram__bytes_read.sum"]
                first_row_write = raw_df.iloc[0]["dram__bytes_write.sum"]

                click.echo(
                    f"Second row values - read: {first_row_read}, write: {first_row_write}"
                )

                # If these are strings and match expected unit formats
                if (
                    isinstance(first_row_read, str)
                    and isinstance(first_row_write, str)
                    and first_row_read in ["Gbyte", "Mbyte", "Kbyte", "byte"]
                    and first_row_write in ["Gbyte", "Mbyte", "Kbyte", "byte"]
                ):
                    read_unit = first_row_read
                    write_unit = first_row_write

                    # Skip the units row for the actual data
                    data_df = raw_df.iloc[1:].reset_index(drop=True)
                    click.echo(f"Found units - read: {read_unit}, write: {write_unit}")
                    click.echo(
                        f"Data DataFrame shape after skipping unit row: {data_df.shape}"
                    )

            # Convert values based on units (sum all values after conversion)
            read_values = pd.to_numeric(
                data_df["dram__bytes_read.sum"], errors="coerce"
            )
            write_values = pd.to_numeric(
                data_df["dram__bytes_write.sum"], errors="coerce"
            )

            # First calculate in GiB
            if read_unit == "Gbyte":
                read_gib = read_values.sum()  # Already in GiB
            elif read_unit == "Mbyte":
                read_gib = read_values.sum() / 1024  # MiB to GiB
            elif read_unit == "Kbyte":
                read_gib = read_values.sum() / (1024 * 1024)  # KiB to GiB
            elif read_unit == "byte":
                read_gib = read_values.sum() / (1024 * 1024 * 1024)  # B to GiB
            else:
                # Default - assume values are already in GiB
                read_gib = read_values.sum()

            if write_unit == "Gbyte":
                write_gib = write_values.sum()  # Already in GiB
            elif write_unit == "Mbyte":
                write_gib = write_values.sum() / 1024  # MiB to GiB
            elif write_unit == "Kbyte":
                write_gib = write_values.sum() / (1024 * 1024)  # KiB to GiB
            elif write_unit == "byte":
                write_gib = write_values.sum() / (1024 * 1024 * 1024)  # B to GiB
            else:
                # Default - assume values are already in GiB
                write_gib = write_values.sum()

            click.echo(
                f"Calculated sums - read_gib: {read_gib}, write_gib: {write_gib}"
            )

            # Divide by the number of profiler replays
            read_gib /= repeats
            write_gib /= repeats

            click.echo(
                f"After repeat adjustment - read_gib: {read_gib}, write_gib: {write_gib}"
            )

            records.append(
                {
                    "file": f.name,
                    "DRAM read [GiB]": read_gib,
                    "DRAM write [GiB]": write_gib,
                    "Total [GiB]": read_gib + write_gib,
                }
            )

        except Exception as e:
            click.echo(f"Error processing {f}: {e}", err=True)
            continue

    if not records:
        click.echo("No usable CSVs → nothing to summarise.", err=True)
        sys.exit(1)

    summary = pd.DataFrame.from_records(records).set_index("file").round(3)

    # -------------------------------------------------------------------------
    # 1) interactive view in the terminal / notebook
    # -------------------------------------------------------------------------
    pd.set_option("display.float_format", "{:,.3f}".format)
    click.echo("\nSummary of all files:")
    click.echo(summary)

    # -------------------------------------------------------------------------
    # 2) write to disk
    # -------------------------------------------------------------------------
    out_file = folder / "summary.csv"
    summary.to_csv(out_file)
    click.echo(f"\n✅  Wrote {out_file}")


# -----------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    main()
