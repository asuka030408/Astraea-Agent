"""Register tools exposed to Astraea."""

from astraea.tools.files import read_file, search_files, write_file
from astraea.tools.commands import run_command
from astraea.tools.memory import search_memory
from astraea.tools.web_search import search_web


TOOLS = [read_file, write_file, search_files, run_command, search_memory, search_web]
