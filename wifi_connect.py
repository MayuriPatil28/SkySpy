import platform
import subprocess

from typing import List


class WiFi:
    def __init__(self) -> None:
        self.os = platform.system()

    def get_wifi_networks(self) -> List[str]:
        try:
            if self.os == "Windows":
                cmd = "netsh wlan show networks"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                output = result.stdout.split(sep="\n")
                ssids = []
                for i in range(len(output)):
                    if output[i].find("SSID") >= 0:
                        ssids.append(output[i][output[i].find(":") + 1 :].strip())
                return ssids
            elif self.os == "Linux":
                # Run nmcli command to list available Wi-Fi networks
                cmd = "nmcli -f 'SSID' device wifi list"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                output = result.stdout

                # Split the output by newlines and filter out empty lines
                networks = output.split("\n")[1:]
                for i in range(len(networks)):
                    networks[i] = networks[i].strip()
                return networks
            elif self.os == "Darwin":
                # Run networksetup command to list available Wi-Fi networks
                cmd = "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/A/Resources/airport scan"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                output = result.stdout

                # Split the output by newlines and filter out empty lines
                networks = output.split("\n")[1:]
                for i in range(len(networks)):
                    networks[i] = networks[i].split("-")[0]
                    networks[i] = networks[i].strip()
                return networks
        except Exception as e:
            print(f"Error: {str(e)}")
            return []

    def get_current_network(self) -> str | None:
        try:
            # Run nmcli command to list available Wi-Fi networks
            if self.os == "Windows":
                cmd = '''netsh wlan show interfaces | findstr /C:"SSID"'''
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                output = result.stdout.split(sep="\n")[0]
                output = output[output.find(":") + 1 :].strip()
                return output
            elif self.os == "Linux":
                cmd = "nmcli -t -f active,ssid dev wifi"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                output = result.stdout

                # Split the output by newlines and filter out empty lines
                networks = output.split("\n")
                for network in networks:
                    if network[0:3] == "yes":
                        return network[4:]
            elif self.os == "Darwin":
                # Run networksetup command to get the current Wi-Fi network
                cmd = "networksetup -getairportnetwork en0 | awk '{print $4}'"
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                output = result.stdout.strip()
                return (
                    output
                    if output
                    != "Current Wi-Fi Network: You are not associated with an AirPort network."
                    else None
                )
        except Exception as e:
            print(f"Error: {str(e)}")
            return None

    def connect_to_wifi(
        self, network_name, password, print_output: bool = False
    ) -> None:
        try:
            if self.os == "Windows":
                print(
                    "Please ensure drone is connected if script is running in Windows. No Automatic connection facility."
                )
            if self.os == "Linux":
                if password:
                    # Connect to the Wi-Fi network with password
                    cmd = f"nmcli device wifi connect '{network_name}' password '{password}'"
                else:
                    # Connect to the Wi-Fi network without password
                    cmd = f"nmcli device wifi connect '{network_name}'"

                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if print_output:
                    print(result.stdout)

            elif self.os == "Darwin":
                if password:
                    # Connect to the Wi-Fi network with password
                    cmd = f"networksetup -setairportnetwork en0 '{network_name}' '{password}'"
                else:
                    # Connect to the Wi-Fi network without password
                    cmd = f"networksetup -setairportnetwork en0 '{network_name}'"

                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                if print_output:
                    print(result.stdout)

        except Exception as e:
            print(f"Error: {str(e)}")