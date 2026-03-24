"""
Run from your text-miners/ folder:
    python3 correct_annotations_v2.py

Revised classifier — built for preprocessed (stop-word-removed) CFPB text.
"""
import re, csv

def classify(narrative, product=""):
    t = narrative.lower()

    # ── HIGH ──────────────────────────────────────────────────────────────
    # Any single keyword from this list → HIGH
    HIGH_KEYWORDS = [
        "garnish", "garnishment", "levied", "bank levy",
        "lawsuit", "filed suit", "filing suit", "sue ", "sued ", "suing ",
        "foreclosure", "eviction", "evicted",
        "sheriff sale", "trustee sale",
        "judgment", "judgement",
        "chapter 7", "chapter 11", "chapter 13", "chapter bankruptcy",
        "attorney filed", "attorney retained", "hired attorney", "retained attorney",
        "scra", "deployed overseas",
        "wage garnish", "wages garnished", "paycheck garnished",
        "homeless", "no food", "cannot eat",
        "suicid",
        "identity theft report", "ftc identity theft", "identity theft affidavit",
        "police report identity", "identity theft police",
        "victim identity theft",
        "filed police report",
    ]
    for kw in HIGH_KEYWORDS:
        if kw in t: return "high"

    # HIGH — regex patterns for preprocessed text
    HIGH_PATTERNS = [
        # Multiple fraudulent accounts opened
        r"(multiple|several|many|numerous).{0,40}(fraudulent|unauthorized).{0,20}account",
        r"(fraudulent|unauthorized).{0,40}account.{0,200}(multiple|several|many|numerous)",
        r"\d{2,}.{0,20}account.{0,30}(opened|name|fraud|without)",
        # Cannot pay / afford (preprocessed versions)
        r"(can.?t|cannot|unable).{0,20}pay.{0,20}rent",
        r"(can.?t|cannot|unable).{0,20}afford",
        r"unable.{0,20}pay",
        r"unable.{0,10}(make|afford).{0,20}payment",
        r"afford.{0,30}(rent|food|bill|mortgage|basic)",
        r"pay.{0,10}rent.{0,30}(cannot|unable|can.?t|afford)",
        r"rent.{0,30}(cannot|unable|can.?t|afford).{0,30}pay",
        # Identity theft + confirmed action (police/ftc/court)
        r"identity.{0,20}theft.{0,300}(police|ftc|court|affidavit|report)",
        r"(police|ftc|court).{0,300}identity.{0,20}theft",
        r"identity.{0,20}stolen.{0,200}(police|ftc|report|account)",
        # Military / SCRA
        r"military.{0,50}(charged|account|loan|card)",
        # Fixed income + cannot afford
        r"fixed.{0,10}income.{0,100}(cannot|afford|unable|struggle)",
        # Confirmed money stolen/lost due to fraud
        r"(money|fund|account).{0,30}stolen.{0,100}(dispute|report|police|fraud)",
        r"account.{0,20}hacked.{0,100}(money|fund|withdraw|stolen)",
        r"hacked.{0,50}account.{0,100}(money|withdraw|stolen|gone)",
        # Child support blocked (HIGH per guide)
        r"child.{0,10}support.{0,80}(blocked|closed|cannot|unable|access|card)",
        # Homeless + family member
        r"homeless.{0,50}(child|family|kid|son|daughter|one)",
    ]
    for p in HIGH_PATTERNS:
        if re.search(p, t): return "high"

    # ── MEDIUM ────────────────────────────────────────────────────────────
    MEDIUM_KEYWORDS = [
        "harassment", "harassed", "harassing",
        "identity theft",
        "unauthorized account", "fraudulent account",
        "unauthorized transaction", "unauthorized charge",
        "fraudulent charge", "fraudulent inquiry",
        "financial hardship", "economic hardship",
        "unemployed", "unemployment",
        "lost job", "lost my job", "losing job",
        "lose job",
        "predatory",
        "denied mortgage", "denied loan", "denied credit",
        "denied apartment", "denied housing",
        "refused mortgage", "refused loan",
        "rejected mortgage", "rejected loan",
        "credit score dropped", "credit score decreased",
        "credit score impacted", "credit score damaged",
        "credit score lowered", "credit score hurt",
        "forbearance", 
        "emotional distress",
        "ruining my credit", "ruining credit",
        "ruining my life",
    ]
    for kw in MEDIUM_KEYWORDS:
        if kw in t: return "medium"

    MEDIUM_PATTERNS = [
        # Denied credit/loan/housing
        r"(denied|rejected|refused).{0,80}(mortgage|loan|credit card|apartment|housing|home|rent)",
        r"denied.{0,30}(home|house|car|vehicle)\b",
        r"lost.{0,20}apartment",
        # Credit score harm
        r"credit.{0,10}(score|rating).{0,80}(drop|fell|decreased|lowered|impacted|damaged|hurt|plummet|point)",
        r"score.{0,30}(drop|fell|decreased|lowered|impacted|damaged|hurt|plummet|went.{0,10}down)",
        r"credit.{0,10}score.{0,100}(point|affect|impact)",
        # Harassment / repeated contact
        r"(constant|repeated|multiple|daily).{0,10}call",
        r"calling.{0,40}(every day|multiple|daily|repeated|again|time)",
        r"call.{0,30}(time|day|week).{0,20}(repeated|multiple|daily|again)",
        r"repeated.{0,20}(phone|call|contact)",
        r"call.{0,10}(every|multiple).{0,10}(day|week|time)",
        # Cannot access funds / account locked
        r"(cannot|can.?t|unable).{0,20}(access|withdraw|use).{0,20}(fund|money|account)",
        r"(fund|money|account).{0,30}(frozen|locked|blocked|restricted|suspended)",
        r"(account|card).{0,20}(locked|blocked|frozen|suspended|restricted)",
        r"locked.{0,20}(account|card|out)",
        # COVID / pandemic hardship
        r"(covid|pandemic|coronavirus).{0,200}(hardship|unable|cannot|struggle|loss|unemployed|job)",
        r"(hardship|unable|cannot|struggle).{0,200}(covid|pandemic|coronavirus)",
        # Job loss
        r"(lost|lose|losing|laid off|layoff).{0,20}(job|employment|income|work)",
        r"(job|employment|income).{0,20}(lost|lose|losing|laid off)",
        r"laid.{0,5}off",
        # Unresolved dispute
        r"dispute.{0,100}(ignored|refused|denied|rejected|unresolved|nothing|no response|not resolved)",
        r"(multiple|several|repeated|numerous).{0,30}(attempt|call|contact|letter|email|time).{0,100}(no response|ignored|refused|nothing|denied|unanswered)",
        r"(no response|no reply|never respond|never called back).{0,100}(dispute|complaint|letter)",
        r"(dispute|complaint).{0,100}(no response|no reply|never respond|ignored|refused)",
        # Forbearance reported late
        r"forbearance.{0,200}(late|delinquent|credit|report|mark)",
        r"(late|delinquent).{0,100}(forbearance|covid|pandemic|hardship)",
        r"reported.{0,30}(late|delinquent).{0,100}(forbearance|covid|pandemic)",
        # Cannot buy home/car due to issue
        r"(cannot|can.?t|unable).{0,30}(buy|purchase|get|obtain).{0,20}(home|house|car|vehicle|mortgage|loan)",
        r"(buying|purchasing|getting).{0,20}(home|house|car).{0,100}(denied|rejected|unable|cannot|can.?t)",
        # Identity theft (any)
        r"identity.{0,20}theft",
        r"identity.{0,20}stolen",
        r"(personal.{0,20}information|ssn|social security).{0,100}(stolen|used|fraud|opened|account)",
        r"(opened|using).{0,20}(account|credit|loan).{0,50}(without|name|permission|consent)",
        # Fraud ongoing
        r"fraud.{0,100}(dispute|claim|report|investigation|unresolved|denied|ignored)",
        r"(account|money|fund|card).{0,50}(fraud|stolen|hacked|unauthorized)",
        r"fraudulent.{0,100}(account|charge|transaction|activity)",
        # Refuse to help
        r"(refuse|refused|refusing).{0,30}(help|investigate|fix|remove|correct|respond)",
        r"(nothing|no one|nobody).{0,30}(help|respond|assist|fix|resolve)",
        # Stimulus / unemployment access issues
        r"(stimulus|eip|unemployment).{0,100}(cannot|access|blocked|unable|missing|waiting|delayed)",
        r"(unemployment|stimulus).{0,100}(fund|card|benefit|payment|check)",
        # Predatory / deceptive
        r"predatory.{0,30}(lend|loan|practice|lending)",
        r"deceptive.{0,30}(practice|advertis|claim|represent)",
        # SBA / business loan denied
        r"sba.{0,80}(denied|rejected|impacted|affected)",
        # Cannot afford basic living (not outright "cannot afford" trigger)
        r"(afford|paying).{0,30}(rent|mortgage|bill|basic|necessit)",
        r"struggling.{0,50}(pay|afford|make payment|financially)",
        r"financial.{0,30}(struggle|difficulty|trouble|problem)",
        # Credit damaged affecting major life decision
        r"credit.{0,50}(affecting|impacting|hurting|damaging).{0,50}(life|home|car|job|mortgage|loan|apartment)",
        r"(home|car|apartment|job|mortgage).{0,50}(denied|rejected|unable|cannot).{0,50}credit",
        # Debt collector violations
        r"(fdcpa|fair debt collection)",
        r"debt.{0,30}collector.{0,100}(harass|illegal|violat|call|contact)",
        # Multiple failed resolution attempts
        r"(called|contacted|written|emailed).{0,30}(multiple|several|many|numerous|repeated).{0,30}time",
        r"(multiple|several|many|numerous).{0,30}(call|time|attempt|letter).{0,100}(unresolved|help|fix|response)",
    ]
    for p in MEDIUM_PATTERNS:
        if re.search(p, t): return "medium"

    return "low"


input_path  = "data/annotation_sample_labelled.csv"
output_path = "data/annotation_sample_labelled_corrected.csv"

with open(input_path, newline="", encoding="utf-8") as fin, \
     open(output_path, "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=["narrative", "product", "risk_label"])
    writer.writeheader()

    changed = 0
    total   = 0
    counts  = {"low": 0, "medium": 0, "high": 0}

    for row in reader:
        total += 1
        old = row.get("risk_label", "").strip()
        new = classify(row["narrative"], row.get("product", ""))
        if old != new:
            changed += 1
        row["risk_label"] = new
        counts[new] += 1
        writer.writerow(row)

print(f"\nDone! Processed {total} rows, changed {changed} labels.")
print(f"\nNew distribution:")
for label in ["low", "medium", "high"]:
    pct = counts[label] / total * 100
    bar = "█" * int(pct / 2)
    print(f"  {label:6s}: {counts[label]:4d}  ({pct:.1f}%)  {bar}")

guide = {"low": (30, 40), "medium": (40, 50), "high": (15, 25)}
print(f"\nGuide targets:")
for label, (lo, hi) in guide.items():
    pct = counts[label] / total * 100
    ok = "✓" if lo <= pct <= hi else f"✗ (target {lo}-{hi}%)"
    print(f"  {label:6s}: {pct:.1f}%  {ok}")

print(f"\nSaved to: {output_path}")
print("\nIf happy, run:")
print("  mv data/annotation_sample_labelled_corrected.csv data/annotation_sample_labelled.csv")