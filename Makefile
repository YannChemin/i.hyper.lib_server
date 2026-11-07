MODULE_TOPDIR = ../..

PGM = i.hyper.lib_server

# The search engine, installed beside the script in etc/i.hyper.lib_server/
# and found there at run time by get_lib_path().
ETCFILES = gaps transforms indexing derive service

include $(MODULE_TOPDIR)/include/Make/Script.make
include $(MODULE_TOPDIR)/include/Make/Python.make
include $(MODULE_TOPDIR)/include/Make/Html.make

# py_compile writes into __pycache__, not beside the source, so no .pyc ever
# appears next to these files to be installed. The engine is imported from
# source, like every other GRASS Python module.
ETCPYCFILES =

default: script html
