target_sources(flextopFD
    PRIVATE
        flextop_FD.cpp
        Hungarian.cpp
        sinkhorn_knopp.cpp
        Hungarian.h
        sinkhorn_knopp.h
        flextop_FD.h
)  

INSTALL(TARGETS flextopFD DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/plugins)