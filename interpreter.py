import sys

from vocab import Vocab
import consts


def _gen_part_list(id: str):
    return f"""<part-list>
        <score-part id="P{id}">
            <part-name>Part</part-name>
            <part-abbreviation>Part</part-abbreviation>
            <score-instrument id="P{id}-I{id}">
                <instrument-name>Part</instrument-name>
            </score-instrument>
        </score-part>
    </part-list>
    """


def _gen_top():
    return '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">'


class Interpreter:
    """Interprets a stream of tokens as MusicXML.

    The job of this is to take a stream of valid tokens and generate valid MusicXML.

    It does this by infering end tags based on the indentation levels of the tokens.
    This works because the limited subset of tokens used in this problem all have a
    unique mapping to some indentation level.

    It also deals with measures, making sure they are properly numbered.
    """

    def __init__(
        self,
        start_token: int,
        vocab: Vocab,
        live_file_out: str = f"{consts.MISC_FILES_ROOT}/live.xml",
    ):
        self.live_file_out = live_file_out
        self.vocab = vocab
        self.start_token = start_token
        self.started = False
        self.prev_interpreted_token = start_token

        # The stack can accumulate multiple tokens. When we're deep in the XML tree
        # and we require one or many tags in a row, we use the stack to dump them out
        # in the correct order.
        self.stack = []

        self.scaled_indents = {}
        for k, v in consts.INDENTS.items():
            self.scaled_indents[k] = v * 4

        self.measure_counter = 0
        self.part_id = 0

        self.file_prev = _gen_top()

    def _std_out(self, s: str):
        sys.stdout.write(s)
        self.file_prev += s

        if self.live_file_out:
            # In order to keep the file constantly updating with valid XML, we need to
            # add the closing tags for the previous token(s) if they were not closed.
            extra = ""
            if self.file_prev.endswith("</note>"):
                extra = "</measure></part></score-partwise>"
            elif self.file_prev.endswith("</measure>"):
                extra = "</part></score-partwise>"
            elif self.file_prev.endswith("</part>"):
                extra = "</score-partwise>"

            if extra and self.live_file_out is not None:
                with open(self.live_file_out, "w") as f:
                    f.write(self.file_prev + extra)

    def _rewrite(self, s: str):
        if s == "<measure>":
            self.measure_counter += 1
            return f'<measure number="{str(self.measure_counter)}">'

        # TODO(jweidinger): I don't think this would really works if
        # there were multiple parts.
        if s == "<part>":
            self.part_id += 1
            self._std_out(_gen_part_list(str(self.part_id)))
            return f'<part id="P{str(self.part_id)}">'

        # TODO(jweidinger): This works for now, but it's not really correct.
        if s == "<beam>":
            return '<beam number="1">'

        return s

    def live_interpret(self, tok_i: int) -> None:
        """Writes to stdout and updates the file at `self.live_file_out` if it is not None.

        Args:
            token: The (next) token to interpret.
        """

        # Special handling for first time this is called purely for cosmetic reasons.
        if not self.started:
            self.started = True
            self.live_interpret(self.start_token)

        token = self.vocab.itos[tok_i]
        if Vocab.is_tag(token):
            indent = self.scaled_indents[token]

            # If we're not going deeper, dump everything that's too deep.
            first_done = False
            while self.stack and indent <= self.stack[-1][1]:
                prev_token, prev_indent = self.stack.pop()
                close_tag = self.vocab.i_to_close_tag[prev_token]
                if first_done:
                    self._std_out(" " * prev_indent)
                self._std_out(close_tag)
                self._std_out("\n")
                first_done = True

            self.stack.append((tok_i, indent))
            if not first_done:
                self._std_out("\n")
            self._std_out(" " * indent)

        self._std_out(self._rewrite(token))

        self.prev_interpreted_token = tok_i
