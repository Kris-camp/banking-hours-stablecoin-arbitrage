"""rob_tcmb_institutions: verify EFT/FAST institutional facts from primary TCMB sources.

Mostly desk research; check (5) re-estimates Table emp_main col (4). The primary sources are pages and
documents on www.tcmb.gov.tr, accessed 2026-10-22 and listed in the JSON; the CBRT Payment Systems
Operational Rule Book, the FAST site and the CBRT annual reports were not used.

Checks:
  (1) Every date in the HOL list of empirics_mech.py should be a day on which TCMB published no
      indicative exchange rate (TCMB FAQ: no indicative rate on official holidays, weekends and
      half working days).
  (2) Every sample weekday that is not in HOL but has no TCMB indicative rate should be a
      half working day (eve of Ramazan/Kurban Bayrami or 28 October). Any other such day is
      reported as unexplained.
  (3) Count the five-minute intervals coded open (bh_eff = 1) that fall after 13:00 Istanbul on
      half working days, when EFT is closed, and the intervals coded closed (bh = 0) that fall
      inside EFT hours 08:30-09:00 and 17:00-17:30 on full business days.
  (4) Convert the FAST per-transaction caps into USD at the sample's own USD/TRY (exp(pg)),
      by month, for each cap regime.

Inputs: data_raw/a5.parquet, data_raw/tcmb_indicative_rate_calendar.csv (HTTP status of
https://www.tcmb.gov.tr/kurlar/YYYYMM/DDMMYYYY.xml for every weekday 2024-01-01..2026-08-31).
Outputs: tables/rob_tcmb_institutions.json, tables/rob_tcmb_institutions.tex.

Notes:
  - check 4: months are restricted to 2024-01..2026-08; the Istanbul-month grouping of exp(pg) would otherwise
    create a 17-interval '2026-09' stub (UTC 21:00-23:55 on 31 Aug 2026).
  - check 3: the headline specifications (empirics_main col 4, empirics_state, empirics_mxn) code the open state
    with bh (clock), not bh_eff; only empirics_mech uses bh_eff. Counts are reported for both codings,
    including weekday-holiday business hours coded open under bh, with the matching denominators.
  - (5) Table emp_main col (4) re-estimated with alternative state codings, to test whether coding errors bias
    the open/closed ratio toward one. Every variant run is reported.
  - fast_participants: the '42 in 2024' count has no primary source; marked UNVERIFIED.
  - paragraph: the 09:00-17:00 window lies inside EFT hours on full business days only (not on half days).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ACCESS = "2026-10-22"

# HOL list copied verbatim from code/empirics_mech.py (weekday public holidays)
HOL = ["2024-01-01", "2024-04-10", "2024-04-11", "2024-04-12", "2024-04-23", "2024-05-01", "2024-06-17", "2024-06-18",
       "2024-06-19", "2024-07-15", "2024-08-30", "2024-10-29",
       "2025-01-01", "2025-03-31", "2025-04-01", "2025-04-23", "2025-05-01", "2025-05-19", "2025-06-06", "2025-06-09",
       "2025-07-15", "2025-10-29",
       "2026-01-01", "2026-03-20", "2026-04-23", "2026-05-01", "2026-05-19", "2026-05-27", "2026-05-28", "2026-05-29",
       "2026-07-15"]
# Half working days (from 13:00) under Law No. 2429: eves (arife) of Ramazan and Kurban Bayrami, and 28 October.
# Weekday ones inside the sample. Kurban arife 2024-06-15 and Ramazan arife 2025-03-29 fall on Saturdays.
HALF = {"2024-04-09": "Ramazan Bayrami eve", "2024-10-28": "Republic Day eve",
        "2025-06-05": "Kurban Bayrami eve", "2025-10-28": "Republic Day eve",
        "2026-03-19": "Ramazan Bayrami eve", "2026-05-26": "Kurban Bayrami eve"}

U = {
    "eft_en": "https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Core+Functions/Payment+Systems/Payment+Systems+in+Turkey/Electronic+Fund+Transfer+System+and+Electronic+Securities+Transfer+System",
    "eft_tr": "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Temel+Faaliyetler/Odeme+Sistemleri/Turkiyedeki+Odeme+Sistemleri/Elektronik+Fon+Transfer+%28EFT%29+Sistemi",
    "pfmi_en": "https://www.tcmb.gov.tr/wps/wcm/connect/31501641-178e-470b-8888-aa91c2958466/FPAiP_UygunlukRaporu_Yay%C4%B1m4_2021_kamuyabeyan-ING.pdf?MOD=AJPERES",
    "pfmi_tr": "https://www.tcmb.gov.tr/wps/wcm/connect/c34c2e8a-803a-4713-9c60-842232677d67/FPAiP_UygunlukRaporu_Yay%C4%B1m4_2021_kamuyabeyan.pdf?MOD=AJPERES",
    "ano2024_12": "https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Announcements/Press+Releases/2024/ANO2024-12",
    "ano2024_16": "https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Announcements/Press+Releases/2024/ANO2024-16",
    "duy2024_16": "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Duyurular/Basin/2024/DUY2024-16",
    "ano2026_36": "https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Announcements/Press+Releases/2026/ANO2026-36",
    "duy2026_36": "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Duyurular/Basin/2026/DUY2026-36",
    "duy2024_08": "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Duyurular/Basin/2024/DUY2024-08",
    "press_lists": "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Duyurular/Basin/{2023,2024,2025,2026}",
    "faq_tr": "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Banka+Hakkinda/Sikca+Sorulan+Sorular",
    "kurlar": "https://www.tcmb.gov.tr/kurlar/YYYYMM/DDMMYYYY.xml",
    "fee_consumer": "https://www.tcmb.gov.tr/wps/wcm/connect/989f1997-5a78-4aa3-99f7-ae816978999d/2023-39+say%C4%B1l%C4%B1+FT+Tebli%C4%9Fi-13+Ocak+2026.pdf?MOD=AJPERES",
    "fee_commercial": "https://www.tcmb.gov.tr/wps/wcm/connect/217dd206-5e42-4402-8fb0-d466b23e5acf/Bankalarca+Ticari+M%C3%BC%C5%9Fterilerden+Al%C4%B1nabilecek+%C3%9Ccretlere+%C4%B0li%C5%9Fkin+Usul+ve+Esaslar+Hakk%C4%B1nda+2020-4+Say%C4%B1l%C4%B1+Tebli%C4%9F.pdf?MOD=AJPERES",
    "crypto_reg": "https://www.tcmb.gov.tr/wps/wcm/connect/6937855a-7c29-4d08-a26e-51ef3273c022/%C3%96demelerde+Kripto+Varl%C4%B1klar%C4%B1n+Kullan%C4%B1lmamas%C4%B1na+Dair+Y%C3%B6netmelik.pdf?MOD=AJPERES",
    "ano2025_34": "https://www.tcmb.gov.tr/wps/wcm/connect/EN/TCMB+EN/Main+Menu/Announcements/Press+Releases/2025/ANO2025-34",
    "ntv2023": "https://www.ntv.com.tr/ntvpara/fast-limiti-50-bin-tlye-yukseltildi,qXsLOiNgZEeG-YeLUnF9vQ",
}

FACTS = [
    {"id": "eft_hours", "status": "VERIFIED (primary)",
     "claim": "The CBRT Payment Systems (interbank EFT, the customer-payment component RPS/POS, and ESTS) open at 08:30 and close at 17:30 on weekdays except public holidays.",
     "sources": [U["eft_en"], U["eft_tr"], U["pfmi_en"], U["pfmi_tr"]],
     "quotes": ["EN page: 'The CBRT Payment Systems operate on week days, except public holidays, between 8:30 a.m. and 5:30 p.m. The systems are open until 1:00 p.m. on half working days.'",
                "TR page: 'TCMB Odeme Sistemleri; resmi tatiller disinda, hafta ici her gun 08.30-17.30 saatleri arasinda calisir. Yarim is gunlerinde ise saat 13.00'e kadar sistemler aciktir.'",
                "PFMI disclosure (29 Dec 2021), s.2.9: 'Except for the official holidays, the systems operate every weekday. In business days, systems open automatically at 08:30 ... Systems close automatically at 17:30 on normal working days and at 13:00 on half working days. The CBRT may change operational days and hours.'"],
     "note": "Web page undated (read 2026-10-22); PFMI disclosure dated 29 Dec 2021. The systems were rebuilt and relaunched on 23 Feb 2024 (press release 2024-12), which says nothing about hours; no TCMB press release in 2023-2026 announces a change in hours (all titles in the 2023-2026 press lists checked). Hours for 2024-2026 are therefore inferred from a 2021 and a 2026 source that agree. The older TCMB glossary figure of 08:00 is superseded."},
    {"id": "eft_components", "status": "VERIFIED (primary)",
     "claim": "EFT has two functional parts: the Turkish Lira Payments System of Customer Payments (RPS, 'POS' in Turkish) and the Turkish Lira Payments System of Interbank Payments (EFT proper). Both run on the same 08:30-17:30 schedule.",
     "sources": [U["eft_en"], U["ano2024_12"], U["pfmi_en"]],
     "quotes": ["EN page: 'Custumer payments are settled in Turkish Lira Payments System of Customer Payments whereas; interbank payments and banks' payments related to international correspondent transactions are settled in Turkish Lira Payments System of Interbank Payments.'",
                "Press release 2024-12 (1 Mar 2024): 'The CBRT Payment Systems, consisting of the Electronic Funds Transfer (EFT) System, Retail Payment System (RPS) and the Electronic Securities Transfer System (ESTS) ... became operational on 23 February 2024 in the redesigned version.'"]},
    {"id": "eft_customer_session", "status": "UNVERIFIED",
     "claim": "Separate (earlier) cut-off for customer payments within the 08:30-17:30 window, and banks' own customer cut-offs.",
     "sources": [U["eft_en"]],
     "quotes": ["EN page: 'However, the Central Bank may change the operating hours to enable banks to make customer payments.' TR: 'bankalar tarafindan musteri odemelerinin gerceklestirilmesi amaciyla, Merkez Bankasi tarafindan farkli calisma saatleri belirlenebilir.'"],
     "note": "TCMB reserves the right to set different hours for customer payments but the public pages give no separate customer-session times. The detailed schedule is in the CBRT Payment Systems Operational Rule Book on eftemkt.tcmb.gov.tr, which was not used, so no customer cut-off time is stated."},
    {"id": "eft_halfday", "status": "VERIFIED (primary) for EFT; half-day calendar from Law 2429 corroborated by TCMB rate files",
     "claim": "On half working days the CBRT Payment Systems close at 13:00. The six weekday half-days in the sample are 2024-04-09, 2024-10-28, 2025-06-05, 2025-10-28, 2026-03-19 and 2026-05-26.",
     "sources": [U["eft_en"], U["pfmi_en"], U["faq_tr"], U["kurlar"]],
     "quotes": ["See eft_hours.", "TCMB FAQ: 'resmi tatiller, hafta sonlari ve yarim gun calisilan gunlerde gosterge kur belirlenmez.' (no indicative rate on official holidays, weekends and half working days)"],
     "note": "TCMB published no indicative rate on any of the six dates (kurlar XML returns 404), as its FAQ says for half working days; see calendar_check. That arife afternoons and 28 October afternoon are half days comes from Law No. 2429, not fetched (mevzuat.gov.tr not tried; not on www.tcmb.gov.tr). That transfers ordered after 13:00 on arife settle on the next business day is reported by the press (e.g. Ekonomist, March 2026) but is UNVERIFIED in a TCMB source."},
    {"id": "eft_unsettled_cancelled", "status": "VERIFIED (primary, 2021 document)",
     "claim": "Payments not settled by the close are cancelled at end of day, not carried over by the CBRT system.",
     "sources": [U["pfmi_en"], U["pfmi_tr"]],
     "quotes": ["RPS: 'Unsettled payments are cancelled at the end of the day.'",
                "EFT: 'Payments that cannot be immediately settled are not rejected but rather queued ... Unsettled payments in the queues are cancelled automatically during end of day processing.'"]},
    {"id": "fast_hours", "status": "VERIFIED (primary)",
     "claim": "FAST is an instant payment system, independent of EFT, that accepts payment orders 24/7; funds are available to the payee within seconds; transactions not completed within the time-out are cancelled.",
     "sources": [U["eft_en"], U["eft_tr"], U["ano2024_16"], U["ano2026_36"]],
     "quotes": ["EN page: 'The Instant and Continuous Transfer of Funds (FAST) System is an immediate payment system where customers can send payment orders 7x24 via system participants. FAST System, designed as an independent system from EFT, is developed, operated, and owned by the CBRT. Payments in the system are executed within seconds, and the funds are instantly available in the recipient's account. Transactions that are not completed within the specified period are canceled.'",
                "Press release 2024-16: FAST 'offers secure, fast, and easy payments around-the-clock'."]},
    {"id": "fast_participants", "status": "VERIFIED (primary) for eligibility and the 2023 count of 36; '42 in 2024' UNVERIFIED",
     "claim": "FAST participants are banks, eligible payment and e-money institutions and PTT; 36 participants after 10 payment/e-money institutions joined in 2023.",
     "sources": [U["eft_tr"], U["duy2024_08"]],
     "quotes": ["TR page: 'Turkiye'de faaliyette bulunan bankalar, gerekli sartlari karsilayan odeme ve elektronik para kuruluslari ile Posta ve Telgraf Teskilati A.S. (PTT) katilimci olabilir.'",
                "DUY2024-08: 'FAST Sistemi katilimci sayisi 2023 yilinda 10 Odeme ve Elektronik Para Kurulusunun sisteme katilimi ile 36'ya cikmistir.'"],
     "note": "Audit: an earlier draft also stated '42 in 2024'; that number appears in neither cited page and is UNVERIFIED."},
    {"id": "fast_limit_pre_apr2024", "status": "VERIFIED (primary) for level; start date 21 Dec 2023 secondary only",
     "claim": "The FAST transaction limit for money transfers was 50,000 TL until 3 April 2024 (merchant FAST-TR QR payments 100,000 TL). It had been raised to 50,000 TL on 21 December 2023.",
     "sources": [U["ano2024_16"], U["duy2024_16"], U["ntv2023"]],
     "quotes": ["Press release 2024-16 (2 Apr 2024): 'as of 4 April 2024, the FAST transaction limit for money transfers will be raised to TRY 100,000 from TRY 50,000, and the limit for merchant payments with dynamic verification using FAST-TR QR Code to TRY 250,000 from TRY 100,000.'",
                "NTV (21 Dec 2023), quoting the TCMB website: '... 21 Aralik 2023 tarihi itibariyle islem basina odemelerin ust limiti 50.000 TL'ye yukseltilmistir.' (secondary)"],
     "note": "No TCMB press release in 2023 or Jan-Mar 2024 concerns FAST limits (press lists checked), so the limit was not changed between 1 Jan and 3 Apr 2024 by press release. The 21 Dec 2023 date was posted on the FAST site (fast.tcmb.gov.tr, blocked) and is UNVERIFIED in a primary source; it lies before the sample in any case."},
    {"id": "fast_limit_apr2024", "status": "VERIFIED (primary)",
     "claim": "From 4 April 2024: 100,000 TL per money transfer; 250,000 TL for merchant payments with dynamic verification using FAST-TR QR Code.",
     "sources": [U["ano2024_16"], U["duy2024_16"]],
     "quotes": ["'As of 4 April 2024, the FAST transaction limits will be raised to TRY 100,000 for money transfers and TRY 250,000 for merchant payments using FAST-TR QR Code.'"],
     "note": "The paper's '250,000 lira for commercial accounts' is wrong: 250,000 TL is the QR merchant-payment limit. TCMB sets the limit per transaction type, not by individual vs commercial account."},
    {"id": "fast_limit_aug2026", "status": "VERIFIED (primary)",
     "claim": "From 26 August 2026: 300,000 TL for money transfers and Request-to-Pay (Odeme Iste) payments, and 300,000 TL for FAST-TR QR merchant payments.",
     "sources": [U["ano2026_36"], U["duy2026_36"]],
     "quotes": ["Press release 2026-36 (24 Aug 2026): 'as of August 26, 2026, the FAST transaction limits for money transfers and payments via the Request-to-Pay Overlay Service will be raised to TRY 300,000 from TRY 100,000, and the limit for merchant payments with dynamic verification using FAST-TR QR Code to TRY 300,000 from TRY 250,000.'"],
     "note": "The 300,000 TL cap covers the last four business days of the sample (26, 27, 28 and 31 August 2026). No other FAST limit change appears in the 2024-2026 press lists."},
    {"id": "fast_limit_per_transaction", "status": "VERIFIED (primary wording) / bank daily limits UNVERIFIED",
     "claim": "The TCMB limit is a transaction-amount limit ('islem tutar limiti'), i.e. per transaction. Whether banks add daily ceilings is not covered by TCMB sources.",
     "sources": [U["duy2024_16"], U["duy2026_36"]],
     "quotes": ["TR title: 'FAST Sistemi Islem Tutar Limitinin Artirilmasi Hakkinda Basin Duyurusu'."]},
    {"id": "after_hours_fx_try", "status": "UNVERIFIED (no TCMB statement found)",
     "claim": "TCMB statement on settlement of FX or TRY transfers ordered outside EFT hours.",
     "sources": [U["eft_en"], U["pfmi_en"], U["faq_tr"], U["press_lists"]],
     "quotes": ["The only TCMB statements found are: EFT/RPS hours and end-of-day cancellation (see above); FAST 7x24 'independent from EFT'; and the FAQ: 'Merkez Bankasi disindaki gercek ve tuzel kisiler arasinda yapilan islemlerde ise, hangi kurun uygulanacagina bu kisilerin kendileri karar verir.'"],
     "note": "No TCMB page or press release (2023-2026) says how banks treat lira or FX transfer orders placed outside EFT hours. Next-business-day execution of after-hours EFT orders is bank practice reported by the press: UNVERIFIED in a primary source. Interbank FX transfers in Turkey do not settle in TCMB systems (TCMB systems settle Turkish lira only: 'The EFT System provides real time transfer and real time gross settlement of Turkish Lira interbank payments'); the FX transfer system operator was not checked: UNVERIFIED."},
    {"id": "fx_indicative_rate_hours", "status": "VERIFIED (primary)",
     "claim": "TCMB sets indicative USD/TRY rates once per business day at 15:30 from bank quotes collected hourly 10:00-15:00; none on weekends, official holidays and half working days.",
     "sources": [U["faq_tr"]],
     "quotes": ["'Gosterge Kurlar; Merkez Bankasi tarafindan her is gunu, saat 15.30'da belirlenir.' 'Oncelikle 10.00-15.00 arasinda her saat basi, bankalararasi doviz piyasasinda 1 ABD dolari karsiliginda Turk lirasina kotasyon veren (fiyat veren) bankalarin alim ve satim fiyat ortalamalasini alir.' 'resmi tatiller, hafta sonlari ve yarim gun calisilan gunlerde gosterge kur belirlenmez.'"]},
    {"id": "fee_no_offhours_surcharge", "status": "VERIFIED (primary) for current texts; pre-Dec-2023 surcharge secondary",
     "claim": "TCMB's consolidated fee communiques in force during the sample cap EFT and FAST fees identically and contain no surcharge for transfers outside EFT hours; FAST fees are subject to the EFT caps.",
     "sources": [U["fee_consumer"], U["fee_commercial"]],
     "quotes": ["Consumer communique 2020/7 as amended by 2023/39 (OG 22/12/2023, No. 32407), Art. 12(6): 'FAST islemleri karsiliginda alinabilecek ucret, bu maddenin ucuncu fikrasinda belirtilen azami ucret sinirlamalarina tabidir.'",
                "Commercial communique 2020/4 (consolidated to 2026/5), Art. 15(2): '(Mulga: RG-25/11/2023-32380)'; Art. 15(7): FAST fees subject to the EFT caps."],
     "note": "Earlier versions (Dec 2020) allowed a 50% higher cap on FAST transfers made outside EFT's official hours; press reports date its removal to 1 December 2023 (secondary). During the sample, the after-hours constraint on lira transfers is the FAST amount cap, not price."},
    {"id": "pi_emi_crypto_ban", "status": "VERIFIED (primary)",
     "claim": "Since 30 April 2021 payment and e-money institutions may not intermediate fund transfers to or from crypto-asset platforms.",
     "sources": [U["crypto_reg"]],
     "quotes": ["Art. 4(2): 'Odeme ve elektronik para kuruluslari, kripto varliklara iliskin alim satim, saklama, transfer veya ihrac hizmeti sunan platformlara veya bu platformlardan yapilacak fon aktarimlarina aracilik edemez.' Art. 5: 'Bu Yonetmelik hukumleri 30/4/2021 tarihinde yururluge girer.' (OG 16 Apr 2021, No. 31456)"]},
    {"id": "papara_2025", "status": "VERIFIED (primary)",
     "claim": "On 27 May 2025 the SDIF (TMSF) was appointed trustee of Papara Elektronik Para A.S.; payment transactions through it became subject to temporary daily limits.",
     "sources": [U["ano2025_34"]],
     "quotes": ["'the Savings Deposit Insurance Fund (SDIF) has been appointed as trustee of Papara Elektronik Para A.S. ... payment transactions provided through the Institution will be temporarily subject to daily limits.'"]},
]

BIB = [
    {"key": "tcmb_eft", "status": "primary", "accessed": ACCESS, "url": U["eft_en"],
     "entry": "Central Bank of the Republic of Turkiye (2026a). Electronic Fund Transfer System - Electronic Securities Transfer System - The Instant and Continuous Transfer of Funds (FAST) System. Payment Systems in Turkiye (web page, undated)."},
    {"key": "tcmb_eft_tr", "status": "primary", "accessed": ACCESS, "url": U["eft_tr"],
     "entry": "Turkiye Cumhuriyet Merkez Bankasi (2026). Elektronik Fon Transfer Sistemi - Elektronik Menkul Kiymet Transfer Sistemi - Fonlarin Anlik ve Surekli Transferi (FAST) Sistemi (web page, undated)."},
    {"key": "tcmb_pfmi2021", "status": "primary", "accessed": ACCESS, "url": U["pfmi_en"],
     "entry": "Central Bank of the Republic of Turkiye (2021). CBRT Payment Systems: Disclosure of the Self-Assessment of Observance of the CPMI-IOSCO Principles for Financial Market Infrastructures, 29 December 2021. Turkish version: " + U["pfmi_tr"]},
    {"key": "tcmb_redesign2024", "status": "primary", "accessed": ACCESS, "url": U["ano2024_12"],
     "entry": "Central Bank of the Republic of Turkiye (2024a). Press Release on the Redesign of the CBRT Payment Systems, No. 2024-12, 1 March 2024."},
    {"key": "tcmb_fast2024", "status": "primary", "accessed": ACCESS, "url": U["ano2024_16"],
     "entry": "Central Bank of the Republic of Turkiye (2024b). Press Release on Increasing FAST System Transaction Limits, No. 2024-16, 2 April 2024. Turkish version: " + U["duy2024_16"]},
    {"key": "tcmb_fast2026", "status": "primary", "accessed": ACCESS, "url": U["ano2026_36"],
     "entry": "Central Bank of the Republic of Turkiye (2026b). Press Release on Increasing FAST System Transaction Limits, No. 2026-36, 24 August 2026. Turkish version: " + U["duy2026_36"]},
    {"key": "tcmb_faq", "status": "primary", "accessed": ACCESS, "url": U["faq_tr"],
     "entry": "Turkiye Cumhuriyet Merkez Bankasi (2026). Sikca Sorulan Sorular: Gosterge niteligindeki kurlar (web page, undated)."},
    {"key": "tcmb_kurlar", "status": "primary (data)", "accessed": ACCESS, "url": U["kurlar"],
     "entry": "Turkiye Cumhuriyet Merkez Bankasi. Indicative exchange rate files, one per business day, probed for every weekday 2 Jan 2024 - 31 Aug 2026."},
    {"key": "tcmb_fee", "status": "primary", "accessed": ACCESS, "url": U["fee_consumer"],
     "entry": "Turkiye Cumhuriyet Merkez Bankasi. Finansal Tuketicilerden Alinacak Ucretlere Iliskin Usul ve Esaslar Hakkinda Teblig (Sayi: 2020/7), consolidated as amended by Teblig 2023/39 (Official Gazette No. 32407, 22 December 2023). Commercial-customer communique 2020/4, consolidated to 2026/5: " + U["fee_commercial"]},
    {"key": "tcmb_crypto2021", "status": "primary", "accessed": ACCESS, "url": U["crypto_reg"],
     "entry": "Turkiye Cumhuriyet Merkez Bankasi (2021). Odemelerde Kripto Varliklarin Kullanilmamasina Dair Yonetmelik. Official Gazette No. 31456, 16 April 2021, effective 30 April 2021."},
    {"key": "tcmb_papara2025", "status": "primary", "accessed": ACCESS, "url": U["ano2025_34"],
     "entry": "Central Bank of the Republic of Turkiye (2025). Press Release on Papara Elektronik Para A.S., No. 2025-34, 27 May 2025."},
    {"key": "ntv2023fast", "status": "UNVERIFIED (secondary; cites TCMB site, not fetched)", "accessed": "not fetched; search snippet 2026-10-22", "url": U["ntv2023"],
     "entry": "NTV (2023). FAST limiti 50 bin TL'ye yukseltildi, 21 December 2023."},
]

BLOCKED = ["eftemkt.tcmb.gov.tr (Operational Rule Book)", "fast.tcmb.gov.tr", "www3.tcmb.gov.tr (annual reports)",
           "odemelerinmerkezi.org", "web.archive.org"]


def ols_fe(d, ycol, cols, fe="month"):
    """Copied from code/empirics_main.py: within-month OLS, day-clustered SEs with finite-sample correction."""
    Y = d[ycol].values.astype(float)
    X = d[cols].values.astype(float)
    if fe:
        gm = d.groupby(fe)
        Y = Y - gm[ycol].transform("mean").values
        X = X - gm[cols].transform("mean").values
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ X.T @ Y; e = Y - X @ b
    codes = pd.factorize(d.day)[0]
    G = np.zeros((codes.max() + 1, X.shape[1])); np.add.at(G, codes, X * e[:, None])
    V = XtXi @ (G.T @ G) @ XtXi
    ng, n, k = codes.max() + 1, len(Y), X.shape[1]
    V *= ng / (ng - 1) * (n - 1) / (n - k)
    return b, V


def ratio_check(a, bh, hol, half_pm, hm):
    """Check (5). Re-estimate Table emp_main col (4), y on q, up, dn and their interactions with a
    state indicator S, month FE, monthly bands from a5.parquet, day-clustered SEs, for each coding of S below.
    Reports kappa+ closed/open, the open-closed difference, the ratio and the delta-method s.e. of the log ratio.
    All codings run are reported."""
    # EFT hours: 08:30-17:30 on full business days, 08:30-13:00 on half days, closed on holidays
    eft = np.where(hol, 0, np.where(half_pm, 0, (hm >= 510) & (hm < 1050))).astype(float)
    codings = [
        ("bh: clock 09:00-17:00 (paper col 4)", bh.astype(float)),
        ("bh*(1-hol)", (bh * (1 - hol)).astype(float)),
        ("bh*(1-hol)*(1-halfday_pm)", (bh * (1 - hol) * (1 - half_pm)).astype(float)),
        ("EFT hours: 08:30-17:30, 13:00 on half days, 0 on holidays", eft),
        ("clock 08:30-17:30", ((hm >= 510) & (hm < 1050)).astype(float)),
        ("clock 08:30-17:00", ((hm >= 510) & (hm < 1020)).astype(float)),
        ("clock 09:00-17:30", ((hm >= 540) & (hm < 1050)).astype(float)),
    ]
    res = {}
    for name, S in codings:
        d = a[["y", "q", "up", "dn", "month", "day"]].copy()
        for c in ["q", "up", "dn"]:
            d[c + "_S"] = d[c].values * S
        b, V = ols_fe(d, "y", ["q", "up", "dn", "q_S", "up_S", "dn_S"])
        kc, ko = b[1], b[1] + b[4]
        g = np.zeros(6); g[1] = 1 / ko - 1 / kc; g[4] = 1 / ko
        res[name] = {"n_open": int(S.sum()), "kappa_up_closed": float(kc), "se_closed": float(np.sqrt(V[1, 1])),
                     "kappa_up_open": float(ko), "se_open": float(np.sqrt(V[1, 1] + V[4, 4] + 2 * V[1, 4])),
                     "diff": float(b[4]), "se_diff": float(np.sqrt(V[4, 4])), "ratio": float(ko / kc),
                     "log_ratio": float(np.log(ko / kc)), "se_log_ratio": float(np.sqrt(g @ V @ g))}
    res["note"] = ("Diagnostic: Table emp_main col (4) under alternative state codings. Tests whether coding errors relative to "
                   "EFT hours bias the open/closed ratio toward one. Holiday correction raises the ratio; adding "
                   "08:30-09:00 and 17:00-17:30 to the open state lowers it.")
    return res


def main():
    out = {"name": "tcmb_institutions", "type": "desk research with primary-source fetching, no estimation",
           "access_date": ACCESS, "blocked_hosts": BLOCKED, "urls": U, "facts": FACTS,
           "bibliography": BIB}

    # ---- calendar check (checks 1 and 2)
    cal = pd.read_csv(ROOT / "data_raw/tcmb_indicative_rate_calendar.csv", dtype=str)
    cal["day"] = pd.to_datetime(cal.day).dt.strftime("%Y-%m-%d")
    no_rate = set(cal.loc[cal.status == "404", "day"])
    failed = sorted(cal.loc[~cal.status.isin(["200", "404"]), "day"])
    hol_with_rate = sorted(set(HOL) - no_rate)
    unexplained = sorted(no_rate - set(HOL) - set(HALF))
    half_with_rate = sorted(set(HALF) - no_rate)
    out["calendar_check"] = {
        "weekdays_probed": int(len(cal)), "probe_failures": failed,
        "weekdays_without_tcmb_rate": int(len(no_rate)),
        "HOL_days": len(HOL), "HOL_days_with_tcmb_rate (should be empty)": hol_with_rate,
        "half_days": HALF, "half_days_with_tcmb_rate (should be empty)": half_with_rate,
        "no_rate_days_not_in_HOL_or_half (unexplained)": unexplained,
    }

    # ---- coding check (check 3; both the bh and the bh_eff codings)
    a = pd.read_parquet(ROOT / "data_raw/a5.parquet")
    ti = a.index.tz_convert("Europe/Istanbul")
    day = pd.Series(ti.strftime("%Y-%m-%d"), index=a.index)
    hm = np.asarray(ti.hour * 60 + ti.minute)
    wk = np.asarray(ti.dayofweek < 5)
    hol = day.isin(HOL).values
    half = day.isin(list(HALF)).values
    bh = a.bh.values
    bh_eff = bh * (1 - hol)
    half_pm = half & (hm >= 13 * 60)
    half_open_coded = int(((bh_eff == 1) & half_pm).sum())       # identical under bh: half days are not in HOL
    hol_open_coded = int(((bh == 1) & hol).sum())
    full_bd = wk & ~hol & ~half
    edge = ((hm >= 8 * 60 + 30) & (hm < 9 * 60)) | ((hm >= 17 * 60) & (hm < 17 * 60 + 30))
    eft_open_coded_closed = int(((bh == 0) & full_bd & edge).sum())
    n_bh1, n_bh0 = float((bh == 1).sum()), float((bh == 0).sum())
    n_be1, n_be0 = float((bh_eff == 1).sum()), float((bh_eff == 0).sum())
    out["coding_check"] = {
        "intervals_total": int(len(a)),
        "intervals_bh_eff1_on_half_days_after_13": half_open_coded,
        "share_of_bh_eff1": half_open_coded / n_be1,
        "intervals_bh0_inside_EFT_hours_0830_0900_1700_1730": eft_open_coded_closed,
        "share_of_bh0": eft_open_coded_closed / n_bh0,
        "share_of_bh_eff0": eft_open_coded_closed / n_be0,
        "bh_coding (empirics_main col 4, empirics_state, empirics_mxn)": {
            "open_intervals": int(n_bh1), "closed_intervals": int(n_bh0),
            "open_coded_EFT_closed_holiday_0900_1700": hol_open_coded,
            "open_coded_EFT_closed_halfday_after_13": half_open_coded,
            "open_coded_EFT_closed_total": hol_open_coded + half_open_coded,
            "share_of_open": (hol_open_coded + half_open_coded) / n_bh1,
            "closed_coded_EFT_open": eft_open_coded_closed,
            "share_of_closed": eft_open_coded_closed / n_bh0},
        "bh_eff_coding (empirics_mech)": {
            "open_intervals": int(n_be1), "closed_intervals": int(n_be0),
            "open_coded_EFT_closed_halfday_after_13": half_open_coded, "share_of_open": half_open_coded / n_be1,
            "closed_coded_EFT_open": eft_open_coded_closed, "share_of_closed": eft_open_coded_closed / n_be0},
        "note": "Five-minute interval timestamps converted to Europe/Istanbul; minute-of-day of the interval label. "
                "bh in a5.parquet equals 1 exactly on Istanbul 09:00-16:55 labels (checked).",
    }

    # ---- (5) Table emp_main col (4) under alternative state codings
    out["audit_ratio_check"] = ratio_check(a, bh, hol, half_pm, hm)

    # ---- USD value of FAST caps (check 4)
    usdtry = np.exp(a.pg).groupby(ti.strftime("%Y-%m")).mean()
    usdtry = usdtry.loc["2024-01":"2026-08"]   # drop the 17-interval '2026-09' stub (UTC 31 Aug evening)
    regimes = [("50000_TL", 50000, "2024-01", "2024-03"), ("100000_TL", 100000, "2024-04", "2026-08"),
               ("300000_TL", 300000, "2026-08", "2026-08")]
    caps = {}
    for name, tl, m0, m1 in regimes:
        r = usdtry.loc[m0:m1]
        usd = tl / r
        caps[name] = {"months": [m0, m1], "usd_min": round(float(usd.min())), "usd_max": round(float(usd.max())),
                      "usdtry_min": round(float(r.min()), 2), "usdtry_max": round(float(r.max()), 2)}
    caps["10000_USDT_in_TL"] = [round(10000 * float(usdtry.min())), round(10000 * float(usdtry.max()))]
    out["fast_cap_usd"] = caps

    # ---- paragraph and corrections
    para = (
        "The lira leg of the arbitrage settles on payment systems owned and operated by the Central Bank of the "
        "Republic of Türkiye. Interbank and customer lira payments go through EFT, whose components open at 08:30 "
        "and close at 17:30 on weekdays, close at 13:00 on half working days such as the eves of religious holidays, "
        "and do not operate on weekends or public holidays. Payments still unsettled at the close are cancelled rather "
        "than carried to the next day. On full business days our 09:00–17:00 business-hours window lies inside EFT "
        "hours. The FAST "
        "instant payment system runs around the clock, independently of EFT, but limits the amount of each money "
        "transfer: 50,000 lira until 3 April 2024, 100,000 lira from 4 April 2024 and 300,000 lira from 26 August "
        "2026. At sample exchange rates the limit was worth about USD 1,600–3,100 until the last week of the "
        "sample, less than one 10,000-USDT trade. Outside EFT hours, the central bank's systems therefore move lira "
        "between banks only through FAST, in transfers of at most a few thousand dollars each."
    )
    out["paragraph_plain"] = para
    out["paragraph_words"] = len(re.findall(r"\S+", para))
    out["corrections_to_paper"] = [
        "Sec. 5.3 (paper.tex l.373): '100,000 lira for individuals and 250,000 lira for commercial accounts in 2025--2026' "
        "is wrong. TCMB sets one money-transfer limit (100,000 TL from 4 Apr 2024 to 25 Aug 2026; 50,000 TL before; "
        "300,000 TL from 26 Aug 2026); 250,000 TL was the FAST-TR QR merchant-payment limit.",
        "EFT hours are 08:30-17:30 (13:00 on half days), not generic 'business hours'; the paper's 09:00-17:00 window "
        "is inside them, so 08:30-09:00 and 17:00-17:30 on full business days are EFT-open but coded closed.",
        "Six weekday half days (EFT closes 13:00) are coded open 13:00-17:00 in both bh and bh_eff.",
        "Audit: the headline specifications (Table emp_main col 4, emp_state, emp_mxn) code the open state with the "
        "clock indicator bh, so weekday-holiday business hours (EFT closed) are also coded open there; see coding_check.",
        "Audit: recoding the state to EFT hours does not raise the col (4) speed ratio. Holiday correction raises it "
        "(4.2 to 5.6), but adding 08:30-09:00 and 17:00-17:30 to the open state lowers it (EFT-hours coding 3.7; "
        "clock 08:30-17:30 2.7); see audit_ratio_check. The 'understates the effect' claim should not be made.",
    ]
    out["suggested_sentence_sec53"] = (
        "In Turkey, large transfers between banks run through EFT, which operates from 08:30 to 17:30 on weekdays, "
        "until 13:00 on half working days, and not on weekends or public holidays; the instant FAST system runs "
        "around the clock but limits each money transfer to 100,000 lira from April 2024 to August 2026, about "
        "USD 2,100–3,100."
    )

    (ROOT / "tables/rob_tcmb_institutions.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # ---- LaTeX: paragraph + bibliography
    tex_para = (para.replace("–", "--").replace("Türkiye", "T\\\"urkiye").replace("50,000", "50{,}000")
                .replace("100,000", "100{,}000").replace("300,000", "300{,}000").replace("10,000", "10{,}000")
                .replace("1,600", "1{,}600").replace("3,100", "3{,}100"))
    tex_para = tex_para.replace("public holidays.", "public holidays \\citep{tcmb_eft, tcmb_pfmi2021}.")
    tex_para = tex_para.replace("to the next day.", "to the next day \\citep{tcmb_pfmi2021}.")
    tex_para = tex_para.replace("26 August 2026.", "26 August 2026 \\citep{tcmb_fast2024, tcmb_fast2026}.")
    bib = r"""
\bibitem[Central Bank of the Republic of T\"urkiye(2026a)]{tcmb_eft}
Central Bank of the Republic of T\"urkiye (2026a). Electronic Fund Transfer System--Electronic Securities Transfer System--The Instant and Continuous Transfer of Funds (FAST) System. \url{%(eft_en)s}, accessed 22 October 2026.

\bibitem[Central Bank of the Republic of T\"urkiye(2021)]{tcmb_pfmi2021}
Central Bank of the Republic of T\"urkiye (2021). CBRT Payment Systems: Disclosure of the Self-Assessment of Observance of the CPMI-IOSCO Principles for Financial Market Infrastructures, 29 December 2021. \url{%(pfmi_en)s}, accessed 22 October 2026.

\bibitem[Central Bank of the Republic of T\"urkiye(2024)]{tcmb_fast2024}
Central Bank of the Republic of T\"urkiye (2024). Press Release on Increasing FAST System Transaction Limits, No.~2024-16, 2 April 2024. \url{%(ano2024_16)s}, accessed 22 October 2026.

\bibitem[Central Bank of the Republic of T\"urkiye(2026b)]{tcmb_fast2026}
Central Bank of the Republic of T\"urkiye (2026b). Press Release on Increasing FAST System Transaction Limits, No.~2026-36, 24 August 2026. \url{%(ano2026_36)s}, accessed 22 October 2026.
""" % {k: v.replace("%", "\\%") for k, v in U.items()}
    tex = ("% rob_tcmb_institutions.tex -- generated by code/rob_tcmb_institutions.py. Desk research; primary TCMB sources\n"
           "% fetched 2026-10-22. See tables/rob_tcmb_institutions.json for quotes and verification status.\n\n"
           "% (A) Institutional paragraph\n" + tex_para + "\n\n% (B) Suggested replacement for the EFT/FAST sentence in Sec. 5.3\n"
           + out["suggested_sentence_sec53"].replace("–", "--").replace("100,000", "100{,}000")
           .replace("2,100", "2{,}100").replace("3,100", "3{,}100")
           + "\n\n% (C) Bibliography entries\n" + bib)
    (ROOT / "tables/rob_tcmb_institutions.tex").write_text(tex)
    print(json.dumps({k: out[k] for k in ["calendar_check", "coding_check", "fast_cap_usd", "paragraph_words"]}, indent=1))


if __name__ == "__main__":
    main()
