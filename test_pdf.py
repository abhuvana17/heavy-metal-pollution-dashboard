from fpdf import FPDF
import datetime

def sanitize(text):
    replacements = {
        '\u2014': '-', '\u2013': '-',
        '\u201c': '"', '\u201d': '"',
        '\u2022': '*',
        '\u00b5': 'u', '\u03bc': 'u',
        '\u2212': '-', '\u00e9': 'e',
    }
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u00b0', 'deg')
    for char, rep in replacements.items():
        text = text.replace(char, rep)
    return text.encode('latin-1', errors='replace').decode('latin-1')

class ReportPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 10, 'HEAVY METAL POLLUTION - ENVIRONMENTAL RISK REPORT',
                  align='C', new_x='LMARGIN', new_y='NEXT')
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(30, 39, 56)
        self.cell(0, 8, sanitize(f'  {title}'), fill=True, new_x='LMARGIN', new_y='NEXT')
        self.ln(1)

    def two_col_row(self, label, value, label_w=55):
        self.set_font('Helvetica', 'B', 9)
        self.cell(label_w, 7, sanitize(label + ':'))
        self.set_font('Helvetica', '', 9)
        # fpdf2 v2.8+: multi_cell(0) means full usable width, NOT remaining
        usable_w = self.w - self.l_margin - self.r_margin
        value_w  = usable_w - label_w
        self.multi_cell(value_w, 7, sanitize(str(value)), new_x='LMARGIN', new_y='NEXT')

pdf = ReportPDF()
pdf.set_margins(12, 18, 12)
pdf.add_page()
pdf.section_title('1. REPORT METADATA')
pdf.two_col_row('Report Generated', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
pdf.two_col_row('Prepared By', 'B.Tech AIML Student')
pdf.two_col_row('Institution', 'Engineering College')
pdf.two_col_row('Dataset Type', 'Synthetic - 10,000 records | NOT real environmental data')
pdf.section_title('2. LOCATION & SAMPLE INFORMATION')
pdf.two_col_row('State', 'Maharashtra')
pdf.two_col_row('District', 'Pune')
pdf.two_col_row('Predicted Risk', 'Moderate')
pdf.multi_cell(0, 6, 'Note: High accuracy is expected because Risk_Level is derived from PLI.')

out = bytes(pdf.output())
print('PDF OK, size:', len(out), 'bytes')
