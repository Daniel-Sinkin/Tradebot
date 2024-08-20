import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import vectorbt as vbt
from fpdf import FPDF


def generate_report(
    portfolio: vbt.Portfolio, report_type: str, output_path: str = None
) -> None:
    """
    Generate a report from a vectorbt portfolio.

    ### Parameters:
    * portfolio: vbt.Portfolio
        * The vectorbt Portfolio object to generate reports from.
    * report_type: str
        * The type of report to generate. Options are 'dash', 'pdf', and 'xlsx'.
    * output_path: str, optional
        * The path where the report should be saved. For 'dash', this is not used.
    """
    if report_type == "dash":
        generate_dash_report(portfolio)
    elif report_type == "pdf":
        if output_path is None:
            raise ValueError("Output path must be specified for PDF reports.")
        generate_pdf_report(portfolio, output_path)
    elif report_type == "xlsx":
        if output_path is None:
            raise ValueError("Output path must be specified for XLSX reports.")
        generate_xlsx_report(portfolio, output_path)
    else:
        raise ValueError(f"Unknown report type: {report_type}")


def generate_dash_report(portfolio: vbt.Portfolio) -> None:
    """
    Generate a Dash report for the portfolio.
    """
    app = dash.Dash(__name__)

    app.layout = html.Div(
        [html.H1("Portfolio Report"), dcc.Graph(figure=portfolio.plot().figure)]
    )

    app.run_server(debug=True)


def generate_pdf_report(portfolio: vbt.Portfolio, output_path: str) -> None:
    """
    Generate a PDF report for the portfolio.

    ### Parameters:
    * portfolio: vbt.Portfolio
        * The vectorbt Portfolio object to generate a PDF report from.
    * output_path: str
        * The path where the PDF report should be saved.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Portfolio Report", ln=True, align="C")

    pdf.output(output_path)


def generate_xlsx_report(portfolio: vbt.Portfolio, output_path: str) -> None:
    """
    Generate an XLSX report for the portfolio.

    ### Parameters:
    * portfolio: vbt.Portfolio
        * The vectorbt Portfolio object to generate an XLSX report from.
    * output_path: str
        * The path where the XLSX report should be saved.
    """
    writer = pd.ExcelWriter(output_path, engine="xlsxwriter")

    portfolio.orders.to_dataframe().to_excel(writer, sheet_name="Orders")
    portfolio.positions.to_dataframe().to_excel(writer, sheet_name="Positions")

    writer.save()


def main() -> None:
    portfolio = vbt.Portfolio(...)
    generate_report(portfolio, "pdf", "portfolio_report.pdf")
    generate_report(portfolio, "xlsx", "portfolio_report.xlsx")
    generate_report(portfolio, "dash")


if __name__ == "__main__":
    main()
