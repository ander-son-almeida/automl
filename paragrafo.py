from docx import Document
from docx.oxml.shared import OxmlElement
from docx.oxml.ns import qn
from docx.shared import RGBColor
import re

def add_rich_paragraph(document, text):
    """
    Adiciona parágrafo com formatação estilo markdown:
    - **negrito** → texto em negrito
    - [link](url) → hiperlink clicável
    
    Args:
        document: Objeto Document
        text (str): Texto com marcações
    """
    paragraph = document.add_paragraph()
    
    # Padrões regex para negrito e links
    bold_pattern = re.compile(r'\*\*(.*?)\*\*')
    link_pattern = re.compile(r'\[(.*?)\]\((.*?)\)')
    
    # Processa o texto em etapas
    remaining_text = text
    
    while remaining_text:
        # Encontra o próximo padrão especial (negrito ou link)
        next_bold = bold_pattern.search(remaining_text)
        next_link = link_pattern.search(remaining_text)
        
        # Determina qual padrão vem primeiro
        next_special = None
        if next_bold and next_link:
            next_special = next_bold if next_bold.start() < next_link.start() else next_link
        elif next_bold:
            next_special = next_bold
        elif next_link:
            next_special = next_link
        
        # Se não há mais padrões, adiciona o texto restante
        if not next_special:
            paragraph.add_run(remaining_text)
            break
        
        # Adiciona texto normal antes do padrão
        if next_special.start() > 0:
            paragraph.add_run(remaining_text[:next_special.start()])
        
        # Processa o padrão encontrado
        if next_special == next_bold:
            # Adiciona texto em negrito
            run = paragraph.add_run(next_special.group(1))
            run.bold = True
        else:
            # Adiciona hiperlink
            link_text = next_special.group(1)
            link_url = next_special.group(2)
            
            run = paragraph.add_run()
            run.text = link_text
            run.font.color.rgb = RGBColor(0, 0, 255)  # Azul
            run.font.underline = True
            
            # Cria o elemento de hiperlink
            r_id = paragraph.part.relate_to(
                link_url, 
                "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
                is_external=True
            )
            
            hyperlink = OxmlElement('w:hyperlink')
            hyperlink.set(qn('r:id'), r_id)
            new_run = OxmlElement('w:r')
            new_run.append(run._r)
            hyperlink.append(new_run)
            paragraph._p.append(hyperlink)
        
        # Atualiza o texto restante
        remaining_text = remaining_text[next_special.end():]

# Exemplo de uso
doc = Document()

texto = "Este é um **texto formatado** com **negrito** e um [link para o Google](https://google.com) no meio."
add_rich_paragraph(doc, texto)

texto2 = "Você pode usar **combinações** como [**link em negrito**](https://exemplo.com) também."
add_rich_paragraph(doc, texto2)

doc.save("documento_formatado.docx")
