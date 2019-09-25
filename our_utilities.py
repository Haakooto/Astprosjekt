def have_internet():
    try:
        import httplib
    except:
        import http.client as httplib

    conn = httplib.HTTPConnection("www.google.com", timeout=5)
    try:
        conn.request("HEAD", "/")
        conn.close()
        return True
    except:
        conn.close()
        return False


def my_imports():
    dirs = [
        "1_the_Rocket_Engine",
        "2_Planetary_Orbits",
        "3_Habitable_zone",
        "4_Onboard_Orientation_Software",
        "5_Satellite_Launch",
        "6_Preparing_for_Landing",
        "7_Landing",
    ]
    return dirs
