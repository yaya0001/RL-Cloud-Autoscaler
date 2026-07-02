import json
import matplotlib.pyplot as plt

with open("results/Experments/exp4_cold_start.json") as f:
    results = json.load(f)

boot_delays = [0,1,3,5,10]

metrics = [
    ("reward","Reward"),
    ("dropped","Dropped Requests"),
    ("cost","Average Active Servers"),
    ("queue_occ","Queue Occupancy")
]
c=5
for metric,title in metrics:

    plt.figure(figsize=(8,5))

    for agent,data in results.items():

        y=[]

        for bd in boot_delays:
            y.append(data[str(bd)][metric]["mean"])

        plt.plot(
            boot_delays,
            y,
            marker="o",
            linewidth=2,
            label=agent.upper()
        )

    plt.xlabel("Boot Delay")
    plt.ylabel(title)
    plt.title(title + " vs Boot Delay")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    c+=1
    plt.savefig(f"results/Experments/plots_ex4/plot_{metric}_Cold_Start.png",dpi=300)
    plt.show()