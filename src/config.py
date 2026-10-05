"""Tunable defaults and the standard CIC-IoT-2023 8-class label grouping."""

SEED = 42

BENIGN_LABEL = "BenignTraffic"
USE_LOG1P = True
LATENT_DIM = 64
NOISE_STD = 0.05          # Gaussian noise injected at the DCAE input (denoising)
MIN_CLASS_N = 30          # "raw" granularity: classes with fewer rows fold into "Other"

# Standard CIC-IoT-2023 8-class grouping (Neto et al. 2023; used by FedShield-IDS etc.)
GROUP_MAP = {
    "BenignTraffic": "Benign",
    **{a: "DDoS" for a in [
        "DDoS-RSTFINFlood", "DDoS-PSHACK_Flood", "DDoS-SYN_Flood", "DDoS-UDP_Flood",
        "DDoS-TCP_Flood", "DDoS-ICMP_Flood", "DDoS-SynonymousIP_Flood",
        "DDoS-ACK_Fragmentation", "DDoS-UDP_Fragmentation", "DDoS-ICMP_Fragmentation",
        "DDoS-SlowLoris", "DDoS-HTTP_Flood"]},
    **{a: "DoS" for a in ["DoS-UDP_Flood", "DoS-SYN_Flood", "DoS-TCP_Flood", "DoS-HTTP_Flood"]},
    **{a: "Mirai" for a in ["Mirai-greeth_flood", "Mirai-greip_flood", "Mirai-udpplain"]},
    **{a: "Recon" for a in ["Recon-PingSweep", "Recon-OSScan", "Recon-PortScan",
                             "VulnerabilityScan", "Recon-HostDiscovery"]},
    **{a: "Spoofing" for a in ["DNS_Spoofing", "MITM-ArpSpoofing"]},
    **{a: "Web" for a in ["SqlInjection", "CommandInjection", "Backdoor_Malware",
                           "Uploading_Attack", "XSS", "BrowserHijacking"]},
    "DictionaryBruteForce": "BruteForce",
}
