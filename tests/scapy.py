from scapy.all import (
    RadioTap,
    Dot11,
    Dot11Elt,
    sendp,
)

def send_beacon(iface, ssid, bssid, channel):
    """Sends a beacon frame."""

    dot11 = Dot11(type=0, subtype=8, addr1='ff:ff:ff:ff:ff:ff', addr2=bssid, addr3=bssid)
    dot11elt = Dot11Elt(ID='SSID', info=ssid)
    dot11elt_rates = Dot11Elt(ID='Rates', info='\x82\x84\x0b\x16') # Basic rates
    dot11elt_dsset = Dot11Elt(ID='DSset', info=chr(channel)) # Channel
    dot11elt_tim = Dot11Elt(ID='TIM', info='\x00\x01\x00\x00') # Traffic indication map
    frame = RadioTap()/dot11/dot11elt/dot11elt_rates/dot11elt_dsset/dot11elt_tim

    sendp(frame, iface=iface, loop=1, inter=0.1, verbose=0) # loop=1 for infinite loop, inter=0.1 for interval

if __name__ == "__main__":
    iface = "wlan0mon" # Replace with your monitor mode interface name
    ssid = "FakeAP" # Replace with your desired SSID
    bssid = "00:11:22:33:44:55" # Replace with your desired BSSID
    channel = 1 # Replace with your desired channel

    try:
        send_beacon(iface, ssid, bssid, channel)
    except KeyboardInterrupt:
        print("Stopped sending beacon frames.")