target_sources(flextopFD
    PRIVATE
        flextop_FD.cpp
        Hungarian.cpp
        Hungarian.h
        flextop_FD.h
)  

INSTALL(TARGETS flextopFD DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/plugins)