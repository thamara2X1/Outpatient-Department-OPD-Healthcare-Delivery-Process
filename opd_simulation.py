"""
Hospital OPD Patient Queue - Discrete-Event Simulation
=======================================================
Two-stage tandem queue:
    Arrival -> [Registration desks] -> [Doctor consultation] -> Exit
"""

import heapq
import random
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# BASE PARAMETERS --------------------
SEED               = 42     # fixes randomness -> reproducible results
N_PATIENTS         = 500    # patients per simulation run
MEAN_INTERARRIVAL  = 4.0    # avg minutes between arrivals (= 15 patients/hour)
N_REG_DESKS        = 2      # registration desks (c1)
MEAN_REG_TIME      = 3.0    # avg minutes to register one patient
N_DOCTORS          = 3      # doctors (c2)
MEAN_CONS_TIME     = 10.0   # avg minutes per consultation


def simulate(seed=SEED, n_patients=N_PATIENTS,
             mean_interarrival=MEAN_INTERARRIVAL,
             n_reg=N_REG_DESKS, mean_reg=MEAN_REG_TIME,
             n_doc=N_DOCTORS, mean_cons=MEAN_CONS_TIME):
    """Run one simulation and return a list of per-patient record dicts."""
    rng = random.Random(seed)

    # ---- 1. Arrival times (Poisson process) ----
    arrivals = []
    t = 0.0
    for _ in range(n_patients):
        t += rng.expovariate(1.0 / mean_interarrival)
        arrivals.append(t)

    # ---- 2. Stage 1: registration desks, served in arrival order ----
    desk_free = [0.0] * n_reg          # min-heap: when each desk frees up
    heapq.heapify(desk_free)
    reg_records = []
    for arr in arrivals:
        earliest_free = heapq.heappop(desk_free)
        reg_start = max(arr, earliest_free)      # cannot start before arriving
        reg_service = rng.expovariate(1.0 / mean_reg)
        reg_end = reg_start + reg_service
        heapq.heappush(desk_free, reg_end)
        reg_records.append((reg_start, reg_end, reg_service))

    # ---- 3. Stage 2: doctors, served in order of finishing registration ----
    order = sorted(range(n_patients), key=lambda i: reg_records[i][1])
    doc_free = [0.0] * n_doc
    heapq.heapify(doc_free)
    cons = {}
    for i in order:
        reg_end = reg_records[i][1]
        earliest_free = heapq.heappop(doc_free)
        cons_start = max(reg_end, earliest_free)
        cons_service = rng.expovariate(1.0 / mean_cons)
        cons_end = cons_start + cons_service
        heapq.heappush(doc_free, cons_end)
        cons[i] = (cons_start, cons_end, cons_service)

    # ---- 4. Build the per-patient dataset ----
    rows = []
    for i in range(n_patients):
        arr = arrivals[i]
        reg_start, reg_end, reg_service = reg_records[i]
        cons_start, cons_end, cons_service = cons[i]
        reg_wait = reg_start - arr
        cons_wait = cons_start - reg_end
        rows.append({
            "Patient_ID": i + 1,
            "Arrival_Time": round(arr, 2),
            "Reg_Wait": round(reg_wait, 2),
            "Registration_Start": round(reg_start, 2),
            "Registration_End": round(reg_end, 2),
            "Reg_Service_Time": round(reg_service, 2),
            "Cons_Wait": round(cons_wait, 2),
            "Consultation_Start": round(cons_start, 2),
            "Consultation_End": round(cons_end, 2),
            "Consultation_Service_Time": round(cons_service, 2),
            "Total_Waiting_Time": round(reg_wait + cons_wait, 2),
            "Time_in_System": round(cons_end - arr, 2),
        })
    return rows


def metrics(rows, n_reg=N_REG_DESKS, n_doc=N_DOCTORS):
    """Compute performance metrics for one simulation run."""
    n = len(rows)
    avg = lambda k: sum(r[k] for r in rows) / n
    # Utilisation = busy server-minutes / available server-minutes
    horizon = max(r["Consultation_End"] for r in rows)
    reg_busy = sum(r["Reg_Service_Time"] for r in rows)
    doc_busy = sum(r["Consultation_Service_Time"] for r in rows)
    return {
        "n": n,
        "avg_reg_wait": avg("Reg_Wait"),
        "avg_doc_wait": avg("Cons_Wait"),
        "avg_total_wait": avg("Total_Waiting_Time"),
        "avg_time_in_system": avg("Time_in_System"),
        "reg_util": reg_busy / (n_reg * horizon),
        "doc_util": doc_busy / (n_doc * horizon),
    }


def show(m, title):
    """Print one scenario's metrics to the console."""
    print(f"\n=== {title} ===")
    print(f"Patients                 : {m['n']}")
    print(f"Avg registration wait    : {m['avg_reg_wait']:.2f} min")
    print(f"Avg doctor wait          : {m['avg_doc_wait']:.2f} min")
    print(f"Avg TOTAL waiting time   : {m['avg_total_wait']:.2f} min")
    print(f"Avg time in system       : {m['avg_time_in_system']:.2f} min")
    print(f"Registration utilisation : {m['reg_util']*100:.1f}%")
    print(f"Doctor utilisation       : {m['doc_util']*100:.1f}%")


def save_csv(rows, path):
    """Write a list of dicts to a CSV file."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

# ============================================================
#  MAIN
# ============================================================
if __name__ == "__main__":

    summary = []      # one row per scenario, for the report's results table

    def record(label, m, arrival_per_hr, n_doc):
        summary.append({
            "Scenario": label,
            "Patients": m["n"],
            "Arrival_Rate_per_hr": round(arrival_per_hr, 1),
            "Doctors": n_doc,
            "Avg_Reg_Wait_min": round(m["avg_reg_wait"], 2),
            "Avg_Doctor_Wait_min": round(m["avg_doc_wait"], 2),
            "Avg_Total_Wait_min": round(m["avg_total_wait"], 2),
            "Avg_Time_in_System_min": round(m["avg_time_in_system"], 2),
            "Reg_Utilisation_pct": round(m["reg_util"] * 100, 1),
            "Doctor_Utilisation_pct": round(m["doc_util"] * 100, 1),
        })

    base_rate = 60.0 / MEAN_INTERARRIVAL       # 15 patients/hour

    # ---------- BASE SCENARIO (this CSV is the deliverable dataset) ----------
    base_rows = simulate()
    base_m = metrics(base_rows)
    show(base_m, f"BASE SCENARIO ({base_rate:.0f} patients/hr, {N_DOCTORS} doctors)")
    save_csv(base_rows, "opd_dataset.csv")
    record("Base", base_m, base_rate, N_DOCTORS)

    # Graph data starts with the base point, so no scenario is duplicated
    # in the summary table while the curves still include 15/hr and 3 doctors.
    A_load     = [base_rate]
    A_totalwait = [base_m["avg_total_wait"]]
    A_docwait   = [base_m["avg_doc_wait"]]
    B_docs     = [N_DOCTORS]
    B_totalwait = [base_m["avg_total_wait"]]

    # ---------- EXPERIMENT A: vary ARRIVAL RATE, doctors fixed at 3 ----------
    # Load is controlled by mean_interarrival: SMALLER = patients arrive
    # faster = higher load. (Raising n_patients does NOT raise load --
    # it only makes the simulation run longer.)
    # 4.0 min is omitted: that is the base scenario, already recorded above.
    for ia in [6, 5, 3.5, 3]:
        m = metrics(simulate(mean_interarrival=ia))
        rate = 60.0 / ia
        show(m, f"EXP A - {rate:.1f} patients/hour")
        record(f"Load {rate:.1f}/hr", m, rate, N_DOCTORS)
        A_load.append(rate)
        A_totalwait.append(m["avg_total_wait"])
        A_docwait.append(m["avg_doc_wait"])

    # ---------- EXPERIMENT B: vary NUMBER OF DOCTORS, load fixed ----------
    # 3 doctors is omitted: that is the base scenario, already recorded above.
    for c in [2, 4, 5]:
        m = metrics(simulate(n_doc=c), n_doc=c)
        show(m, f"EXP B - {c} doctors")
        record(f"{c} doctors", m, base_rate, c)
        B_docs.append(c)
        B_totalwait.append(m["avg_total_wait"])

    # Sort graph points into ascending order
    A = sorted(zip(A_load, A_totalwait, A_docwait))
    A_load, A_totalwait, A_docwait = [list(x) for x in zip(*A)]
    B = sorted(zip(B_docs, B_totalwait))
    B_docs, B_totalwait = [list(x) for x in zip(*B)]

    # ---------- SAVE SUMMARY TABLE ----------
    with open("opd_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("\n================ SUMMARY TABLE ================")
    print(f"{'Scenario':<14}{'Arr/hr':>8}{'Doc':>5}{'TotWait':>10}"
          f"{'DocWait':>10}{'DocUtil%':>10}")
    for r in summary:
        print(f"{r['Scenario']:<14}{r['Arrival_Rate_per_hr']:>8}{r['Doctors']:>5}"
              f"{r['Avg_Total_Wait_min']:>10}{r['Avg_Doctor_Wait_min']:>10}"
              f"{r['Doctor_Utilisation_pct']:>10}")

    # ---------- GRAPHS ----------
    plt.figure(figsize=(7, 4.5))
    plt.plot(A_load, A_totalwait, "o-",  label="Total waiting time")
    plt.plot(A_load, A_docwait,   "s--", label="Doctor-stage wait")
    plt.xlabel("Arrival rate (patients / hour)")
    plt.ylabel("Average waiting time (minutes)")
    plt.title("Experiment A: Waiting Time vs Arrival Rate")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("graph1_arrival_rate_vs_wait.png", dpi=130); plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(B_docs, B_totalwait, "o-", color="green")
    plt.xlabel("Number of doctors")
    plt.ylabel("Average total waiting time (minutes)")
    plt.title("Experiment B: Waiting Time vs Doctor Capacity")
    plt.xticks(B_docs); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("graph2_doctors_vs_wait.png", dpi=130); plt.close()

    plt.figure(figsize=(6, 4.5))
    utils = [base_m["reg_util"] * 100, base_m["doc_util"] * 100]
    bars = plt.bar(["Registration\ndesks", "Doctors"], utils,
                   color=["#6c8ebf", "#b85450"])
    for b, u in zip(bars, utils):
        plt.text(b.get_x() + b.get_width()/2, u + 1, f"{u:.0f}%", ha="center")
    plt.ylabel("Utilisation (%)"); plt.ylim(0, 100)
    plt.title("Base Scenario: Resource Utilisation by Stage")
    plt.tight_layout()
    plt.savefig("graph3_utilisation_by_stage.png", dpi=130); plt.close()

    plt.figure(figsize=(7, 4.5))
    waits = [r["Total_Waiting_Time"] for r in base_rows]
    plt.hist(waits, bins=30, color="#9673a6", edgecolor="white")
    plt.xlabel("Total waiting time (minutes)")
    plt.ylabel("Number of patients")
    plt.title("Base Scenario: Distribution of Patient Waiting Times")
    plt.tight_layout()
    plt.savefig("graph4_wait_distribution.png", dpi=130); plt.close()

    print("\nSaved: opd_dataset.csv, opd_summary.csv, graph1..graph4 PNGs")
