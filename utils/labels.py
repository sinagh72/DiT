def get_merged_classes():
    kermany_classes = {"NORMAL": 0,
                       "DRUSEN": 1,
                       "DME": 2,
                       "CNV": 3,
                       }

    srinivasan_classes = {"NORMAL": 0,
                          "AMD": 1,
                          "DME": 2,
                          }

    oct500_classes = {"NORMAL": 0,
                      "AMD": 1,
                      "DR": 2,
                      "OTHERS": 3,
                      }

    nur_classes = {"NORMAL": 0,
                   "DRUSEN": 1,
                   "CNV": 2,
                   }

    waterloo_classes = {"NORMAL": 0,
                        "AMD": 1,
                        "DR": 2,
                        "HR": 3,
                        "CSR": 4
                        }

    octdl_classes = {"NORMAL": 0,
                     "AMD": 1,
                     "DME": 2,
                     "ERM": 3,
                     "RAO": 4,
                     "RVO": 5,
                     "VID": 6
                     }
    uic_dr_classes = {"Control": 0,
                      "Mild": 1,
                      "Moderate": 2,
                      "Severe": 3,
                      }
    mario_classes = {"Reduced": 0,
                     "Stable": 1,
                     "Increased": 2,
                     "uninterpretable": 3,
                     "None": -1
                     }
    oimhs_classes = {"1": 0,
                     "2": 1,
                     "3": 2,
                     "4": 3,
                     }
    
    thoct_classes = {"NORMAL": 0,
                    "AMD": 1,
                    "DME": 2,
                    }

    return (kermany_classes, srinivasan_classes, oct500_classes, nur_classes, waterloo_classes, octdl_classes,
            uic_dr_classes, mario_classes, oimhs_classes, thoct_classes)


def get_full_classes():
    kermany_classes = {"NORMAL": 0,
                       "DRUSEN": 1,
                       "DME": 2,
                       "CNV": 3,
                       }

    srinivasan_classes = {"NORMAL": 0,
                          "AMD": 1,
                          "DME": 2,
                          }

    oct500_classes = {"NORMAL": 0,
                      "AMD": 1,
                      "DR": 2,
                      "CNV": 3,
                      "OTHERS": 4,
                      "RVO": 5,
                      "CSC": 6,
                      }

    nur_classes = {"NORMAL": 0,
                   "DRUSEN": 1,
                   "CNV": 2,
                   }

    waterloo_classes = {"NORMAL": 0,
                        "AMD": 1,
                        "DR": 2,
                        "HR": 3,
                        "CSR": 4
                        }

    octdl_classes = {"NORMAL": 0,
                     "AMD": 1,
                     "DME": 2,
                     "ERM": 3,
                     "RAO": 4,
                     "RVO": 5,
                     "VID": 6
                     }
    uic_dr_classes = {"Control": 0,
                      "Mild": 1,
                      "Moderate": 2,
                      "Severe": 3,
                      }
    mario_classes = {"Reduced": 0,
                     "Stable": 1,
                     "Increased": 2,
                     "Uninterpretable": 3,
                     }

    oimhs_classes = {"1": 0,
                     "2": 1,
                     "3": 2,
                     "4": 3,
                     }
    thoct_classes = {"NORMAL": 0,
                     "AMD": 1,
                     "DME": 2,
                     }
    return (kermany_classes, srinivasan_classes, oct500_classes, nur_classes, waterloo_classes, octdl_classes,
            uic_dr_classes, mario_classes, oimhs_classes, thoct_classes)


def get_diffusion_classes(): #19 different categories
    kermany_classes = {"NORMAL": 0,
                       "DRUSEN": 1,
                       "DME": 2,
                       "CNV": 3,
                       }

    srinivasan_classes = {"NORMAL": 0,
                          "AMD": 4,
                          "DME": 2,
                          }

    oct500_classes = {"NORMAL": 0,
                      "AMD": 4,
                      "DR": 5,
                      "CNV": 3,
                      "OTHERS": 6,
                      "RVO": 7,
                      "CSC": 8,
                      }

    nur_classes = {"NORMAL": 0,
                   "DRUSEN": 1,
                   "CNV": 3,
                   }

    waterloo_classes = {"NORMAL": 0,
                        "AMD": 4,
                        "DR": 5,
                        "HR": 9,
                        "CSR": 10
                        }

    octdl_classes = {"NORMAL": 0,
                     "AMD": 4,
                     "DME": 2,
                     "ERM": 11,
                     "RAO": 12,
                     "RVO": 7,
                     "VID": 13
                     }
    uic_dr_classes = {"Control": 0,
                      "Mild": 5,
                      "Moderate": 5,
                      "Severe": 5,
                      }
    mario_classes = {"Reduced": 6,
                     "Stable": 6,
                     "Increased": 6,
                     "Uninterpretable": 6,
                     "None": 6
                     }

    oimhs_classes = {"1": 0,
                     "2": 91,
                     "3": 92,
                     "4": 93,
                     }
    thoct_classes = {"NORMAL": 0,
                     "AMD": 4,
                     "DME": 2,
                     }
    return (kermany_classes, srinivasan_classes, oct500_classes, nur_classes, waterloo_classes, octdl_classes,
            uic_dr_classes, mario_classes, oimhs_classes, thoct_classes)
