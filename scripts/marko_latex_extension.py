# pyright: strict

import io
import math
import re
import sys
from typing import Any, NamedTuple, cast

import marko
import marko.block
import marko.ext.gfm as MarkoGFM
import marko.inline
import yaml
from format_python import format_python
from marko import MarkoExtension, block, inline
from marko.ext.latex_renderer import LatexRenderer
from marko.source import Source
from urllib.parse import urlparse, parse_qs

# https://github.com/python/typeshed/issues/3049
if isinstance(sys.stdin, io.TextIOWrapper) and sys.version_info >= (3, 7):
    sys.stdin.reconfigure(encoding="utf-8-sig")

class BlockElementWithPattern(block.BlockElement):
    priority=100
    pattern: re.Pattern[str] | str | None = None
    include_children=False
   
    def __init__(self, match: re.Match[str]) -> None:
        self.content = match.group(1)

    @classmethod
    def match(cls, source: Source) -> re.Match[str] | None:
        if cls.pattern is None:
            raise Exception('pattern not set')
        return source.expect_re(cls.pattern)

    @classmethod
    def parse(cls, source: Source) -> Any:
        m = source.match
        source.consume()
        return m

class BlockMathInParagraph(inline.InlineElement):
    priority=101
    pattern = r'\$\$([\s\S]*?)\$\$'
    parse_children = False
   
    def __init__(self, match: re.Match[str]) -> None:
        self.content = match.group(1)
    
class InlineMath(inline.InlineElement):
    priority=100
    pattern = r'\$([\s\S]*?)\$'
    parse_children = False
    
    def __init__(self, match: re.Match[str]) -> None:
        self.content = match.group(1)
        
class BlockMath(BlockElementWithPattern):
    pattern=re.compile(r'\$\$([\s\S]*?)\$\$', flags=re.M)
    
class FrontMatter(BlockElementWithPattern):
    priority=100
    pattern = re.compile(r'---\n(.*?)\n---\n', re.M | re.DOTALL)
    parse_children = False
    def __init__(self, match: re.Match[str]) -> None:
        super().__init__(match)
        self.data = yaml.safe_load(self.content)
        
class LatexTabular(BlockElementWithPattern):
    pattern = re.compile(r'(\\begin\{tabular\}[\s\S]*\\end\{tabular\})', re.M)
    
class LatexTabularx(BlockElementWithPattern):
    pattern = re.compile(r'(\\begin\{tabularx\}[\s\S]*\\end\{tabularx\})', re.M)
    
class LatexLongTable(BlockElementWithPattern):
    pattern = re.compile(r'(\\begin\{longtable\}[\s\S]*\\end\{longtable\})', re.M)
    
class LatexMinipage(BlockElementWithPattern):
    pattern = re.compile(r'(\\begin\{minipage\}[\s\S]*\\end\{minipage\})', re.M)
    
class CustomFootnote(inline.InlineElement):
    pattern=r'\[\{(.*)\}\]'
    parse_children = False
    
    def __init__(self, match: re.Match[str]) -> None:
        self.content = match.group(1)
        
class Strikethrough(inline.InlineElement):
    pattern=r'\~\~(.*)\~\~'
    parse_children = False
    
    def __init__(self, match: re.Match[str]) -> None:
        self.content = match.group(1)
        
class Emoji(inline.InlineElement):
    pattern=r'\:(.*)\:'
    parse_children = False
    
    def __init__(self, match: re.Match[str]) -> None:
        self.emoji_name = match.group(1)
        
class InterviewQA(inline.InlineElement):
    pattern=r'([QA])\: '
    parse_children = False
    def __init__(self, match: re.Match[str]) -> None:
        self.type = match.group(1)
        

class MarkoLatexRenderer(LatexRenderer):
    front_matter: dict[str, Any] = {}
    
    def render_document(self, element: marko.block.Document):
        children = self.render_children(element)
        
        layout = self.front_matter.get('layout')

        if type(layout) != str:
            raise Exception('layout is unset')

        meta = cast(dict[str, str], self.front_matter.get('meta'))
        return self._environment2(
            layout,
            children,
            meta
        )
    
    def render_heading(self, element: marko.block.Heading):
        """
        Override to get the artile name from the H1 heading.
        """
        if element.level == 1:
            children = self.render_children(element)
            self.article_name = children
            return ""

        # ignore since we can not type the super class _directly_.
        return super().render_heading(element)  # pyright: ignore
    
    def render_fenced_code(self, element: marko.block.FencedCode):
        language = self._escape_latex(element.lang).strip().lower()
        if 'c++' in language or 'cpp' in language:
            language = 'cpp'
        if 'py' in language or 'python' in language:
            language = 'python'
        if language not in ['c', 'cpp', 'python', 'text']:
            language = 'text'

        # This cast got from the marko source code (marko.block.FencedCode#__init__)
        content: str = cast(marko.inline.RawText, element.children[0]).children
        if language == 'python':
            content = format_python(content)
        return self._environment(f"{language}code", content)
    
    def render_block_math(self, element: BlockMath):
        # print('block math', element.content)
        return f"$${element.content}$$"
    
    def render_block_math_in_paragraph(self, element: BlockMathInParagraph):
        # print('block math in paragraph', element.content)
        return f"$${element.content}$$"
    
    def render_inline_math(self, element: InlineMath):
        # print('inline math', element.content)
        return f"${element.content}$"
    
    def render_link(self, element: marko.inline.Link):
        if element.title:
            print("Setting a title for links is not supported!")
        body = self.render_children(element)
        # return f"\\href{{{element.dest}}}{{{body}}} \\footnote{{{self._escape_latex(element.dest)}}}"
        return f"\\insertLink{{ {self._escape_latex(element.dest)} }}{{ {body} }}"
    
    def render_list(self, element: marko.block.List):
        children = self.render_children(element)
        env = "enumerate" if element.ordered else "itemize"
        # TODO: check how to handle element.start with ordered list
        if element.start and element.start != 1:
            print("Setting the starting number of the list is not supported!")
        return self._environment(env, children, ['leftmargin=0.5cm', 'itemsep=1mm', 'topsep=0mm', 'partopsep=0mm', 'parsep=0mm'])
            
    def render_image(self, element: marko.inline.Image):
        children = self.render_children(element)
        parsed_url = urlparse(element.dest)
        options = ''

        if parsed_url.query != '':
            qs = parse_qs(parsed_url.query)
            options = '[\n' + ',\n'.join(f'  {key}={",".join(value)}' for key, value in qs.items()) + '\n]'
        
        return f"\\includeImage{options}{{ {parsed_url.path} }}{{ {children} }}"
        
    def render_custom_footnote(self, element: CustomFootnote):
        return f"\\footnote{{ {element.content} }}"
    
    def render_strikethrough(self, element: Strikethrough):
        return f"\\sout{{ {element.content} }}"
    
    def render_html_block(self, element: marko.block.HTMLBlock):
        print("Rendering HTML is not supported!")
        print(element.children)
        return ""
    
    def render_latex_tabular(self, element: LatexTabular):
        return r'\begin{center}' + element.content + r'\end{center}'
    
    def render_latex_tabularx(self, element: LatexTabularx):
        return r'\begin{center}' + element.content + r'\end{center}'
    
    def render_latex_long_table(self, element: LatexLongTable):
        return r'\begin{center}' + element.content + r'\end{center}'
    
    def render_latex_minipage(self, element: LatexMinipage):
        return r'\begin{center}' + element.content + r'\end{center}'
    
    def render_emoji(self, element: Emoji):
        return f'\\{element.emoji_name}'
    
    def render_line_break(self, element: Any):
        # always soft
        return '\n'
    
    def render_front_matter(self, element: FrontMatter):
        self.front_matter = element.data
        return ''
    
    def render_interview_qa(self, element: InterviewQA):
        if self.front_matter.get('layout') == 'interview':
            return r'\interview' + element.type + ' '
        return element.type + ': '
    
    def render_table(self, element: MarkoGFM.elements.Table):
        casted_children = cast(list[MarkoGFM.elements.TableRow], element.children)
        all_cells = [[cast(MarkoGFM.elements.TableCell, cell) for cell in row.children] for row in casted_children]

        rendered_content = [
            [self.render(cell) for cell in row] for row in all_cells
        ]

        header_cells = all_cells[0]
        
        alignment = self._render_table_alignment(header_cells, rendered_content)
        lines: list[str] = []
        lines.append('\\begin{center}')
        # lines.append(f'\\begin{{tabularx}}{{\\linewidth}}{{ {alignment} }}')
        lines.append(f'\\begin{{tabular}}{{ {alignment} }}')
        lines.append(r'\hline')
        
        def add_row(row: list[str]):
            rendered_row = ' & '.join(row)
            lines.append('  ' + rendered_row + r' \tabularnewline \hline')

        add_row(rendered_content[0])
        lines.append(r'\hline')
        for row in rendered_content[1:]:
            add_row(row)

        # lines.append('\\end{tabularx}')
        lines.append('\\end{tabular}')
        lines.append('\\end{center}')

        return '\n'.join(lines)

    def render_table_row(self, element: MarkoGFM.elements.TableRow):
        return ' & '.join(map(self.render, element.children)) + r' \tabularnewline \hline'

    def _render_table_alignment(self, header_cells: list[MarkoGFM.elements.TableCell], content: list[list[str]]):
        n = len(content)
        m = len(content[0])
        col_max_len = [
            max(len(content[i][j]) for i in range(n)) for j in range(m)
        ]

        lg_max_len = list(map(lambda x: math.log(x) ** 2, col_max_len))
        sum_lg_max_len = sum(lg_max_len)

        fixed_total_size = 0.95

        def map_alignment(alignment: str | None, pos: int):
            ratio = lg_max_len[pos] / sum_lg_max_len
            size = fr'p{{{ratio * fixed_total_size:.2f}\linewidth}}'
            if alignment == 'left':
                return size
            elif alignment == 'right':
                return r'>{\raggedleft\arraybackslash}' + size
            else:
                if alignment != 'center' and alignment is not None:
                    print(f'Warning: Unknown alignment {alignment}. Fall back to "center".')
                return r'>{\centering\arraybackslash}' + size

        return '|' + '|'.join(
            map_alignment(cell.align, pos) for cell, pos in zip(header_cells, range(m))
          ) + '|'

    def render_table_cell(self, element: MarkoGFM.elements.TableCell):
        return self.render_children(element)
    
    @staticmethod
    def _escape_latex(text: str) -> str:
        # print('escaping', text)
        # Special LaTeX Character:  # $ % ^ & _ { } ~ \
        specials = {
            "#": "\\#",
            "$": "\\$",
            "%": "\\%",
            "&": "\\&",
            "_": "\\_",
            "{": "\\{",
            "}": "\\}",
            "^": "\\^{}",
            "~": "\\~{}",
            "\\": "\\textbackslash{}",
            "\"": "''"
        }

        return "".join(specials.get(s, s) for s in text)
    
    @staticmethod
    def _environment2(env_name: str, content: str, options: dict[str,  str] = {}) -> str:
        options_str = '\n'.join(map(lambda item: f'  {item[0]}={{{item[1]}}},', options.items()))
        return f"\\begin{{{env_name}}}[\n{options_str}\n]\n{content}\\end{{{env_name}}}\n"

def make_extension():
    return MarkoExtension(
        elements=[
                BlockMath,
                BlockMathInParagraph,
                InlineMath,
                CustomFootnote,
                Strikethrough,
                LatexTabular,
                LatexLongTable,
                LatexMinipage,
                LatexTabularx,
                Emoji,
                FrontMatter,
                InterviewQA,
                MarkoGFM.elements.Table,
                MarkoGFM.elements.TableRow,
                MarkoGFM.elements.TableCell,
            ],
        renderer_mixins = [MarkoLatexRenderer]
    )
