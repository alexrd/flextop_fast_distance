target_sources(TestFlextopFD
    PRIVATE
        test_flextop.cpp
)  

INSTALL(TARGETS TestFlextopFD DESTINATION "${CMAKE_INSTALL_PREFIX}/tests")