"""
review_data.py
──────────────
Realistic e-commerce review dataset for three smartphones.
Designed to produce varied ABSA outputs — mixed reviews,
contrast sentences, aspect-specific praise/criticism.
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class RawReview:
    review_id:   str
    product_id:  str
    reviewer:    str
    title:       str
    body:        str
    star_rating: int
    verified:    bool
    helpful:     int


PRODUCTS = {
    "P001": "NovaTech X12 Pro",
    "P002": "Lumina S8 Ultra",
    "P003": "SwiftPhone 7",
}

REVIEWS: List[RawReview] = [

    # ─────────────────────────────────────────────
    # P001: NovaTech X12 Pro  (mixed product — good camera, bad battery)
    # ─────────────────────────────────────────────
    RawReview("R001","P001","Arjun M.","Incredible camera, terrible battery",
        "The camera on this phone is absolutely incredible. Photos are sharp, crisp and the night mode "
        "is stunning — easily the best I've used. However, the battery life is terrible. It drains so "
        "fast that I have to charge it twice a day. The display is vibrant and bright, which I love, "
        "but that probably explains the battery drain. Build quality feels premium and solid.", 3, True, 142),

    RawReview("R002","P001","Shalini R.","Great display but software needs work",
        "The screen resolution is excellent and the display is really bright even in sunlight. "
        "Touch response is smooth and fast. Unfortunately the software has too much bloatware "
        "and the UI is not clean at all. There are annoying bugs that cause the phone to freeze "
        "occasionally. Performance is decent for everyday tasks but gaming causes heavy heating. "
        "Battery lasts about a day with moderate use which is acceptable.", 3, True, 87),

    RawReview("R003","P001","Ramesh K.","Best camera phone in this price range",
        "I bought this specifically for the camera and I was not disappointed. The photos are "
        "exceptionally sharp with accurate colors. Video recording is smooth at 4K. The zoom "
        "capability is impressive. Build quality is solid with a premium glass finish. "
        "The display has excellent color accuracy and is bright. Battery could be better "
        "but it is not the worst I have used. Overall great value.", 4, True, 203),

    RawReview("R004","P001","Priya T.","Overheating issue ruins the experience",
        "The performance starts fast but the processor heats up within 20 minutes of gaming. "
        "The phone becomes too hot to hold comfortably. This is a serious issue. Camera is good "
        "for everyday shots but not outstanding. The battery drains extremely fast when the "
        "phone is hot. Software crashes randomly which is very frustrating. Build feels cheap "
        "for a phone at this price point. Disappointed overall.", 2, True, 178),

    RawReview("R005","P001","Kavya S.","Perfect for photography enthusiasts",
        "As someone who loves photography, this phone is brilliant. The camera captures "
        "stunning portraits with beautiful bokeh. Low light shots are clear and noise-free. "
        "The display colors are vibrant and accurate — perfect for editing photos. "
        "Software is stable after the latest update. Build quality is excellent. "
        "Only complaint is the battery life is short — I wish it lasted longer.", 5, True, 312),

    RawReview("R006","P001","Deepak N.","Decent phone with one major flaw",
        "Everything about this phone is good except the battery. The camera is great, "
        "display is bright, performance is smooth for daily use. But the battery is a disaster. "
        "It barely lasts 6 hours on a single charge. Fast charging is good but you need "
        "to carry a charger everywhere. Would not recommend if battery life matters to you.", 3, False, 65),

    RawReview("R007","P001","Sneha V.","Surprised by the audio quality",
        "The speakers on this phone are surprisingly loud and clear with good bass. "
        "Call quality is excellent. The camera is impressive for the price. "
        "Battery life is acceptable, lasts about a day. Display is vibrant. "
        "Build quality feels premium. Happy with my purchase overall.", 4, True, 41),

    RawReview("R008","P001","Vikram L.","Software updates broke my phone",
        "Before the update everything was great. After the latest software update, "
        "the phone lags constantly and crashes apps. Performance has become terrible. "
        "The camera quality also degraded after the update — photos look grainy now. "
        "Battery drain increased dramatically. This is unacceptable. "
        "Please rollback or fix the update.", 1, True, 289),

    # ─────────────────────────────────────────────
    # P002: Lumina S8 Ultra  (premium product — mostly positive)
    # ─────────────────────────────────────────────
    RawReview("R009","P002","Ananya B.","Worth every rupee",
        "The Lumina S8 Ultra is exceptional in every way. The display is absolutely stunning — "
        "the most vibrant and clear screen I have ever seen. Camera quality is outstanding with "
        "incredible detail in every shot. Performance is blazing fast with no lag whatsoever. "
        "Battery life is excellent — easily lasts two full days. Build quality is premium and "
        "the phone feels incredibly solid. Software is clean and smooth. Highly recommend.", 5, True, 421),

    RawReview("R010","P002","Rohit S.","Premium price but justifies it",
        "Yes it is expensive. But the performance is absolutely outstanding. Gaming is perfectly "
        "smooth with zero heating issues. The display is bright with excellent refresh rate. "
        "Camera captures perfect shots even in low light. Battery life is impressive — "
        "I get through a full day with heavy use. Build quality feels incredibly premium. "
        "Software is clean with no bloatware. Worth the price.", 5, True, 198),

    RawReview("R011","P002","Meera P.","Great phone but overpriced",
        "The camera is brilliant and the display is stunning. Performance is fast and reliable. "
        "Software is smooth and clean. Battery life is good — lasts all day. "
        "However I feel the phone is overpriced for what it offers compared to competitors. "
        "Build quality is excellent though. The connectivity is also great — strong wifi signal "
        "and bluetooth range is impressive. Good but not worth the premium price.", 4, True, 156),

    RawReview("R012","P002","Suresh G.","Audio experience is phenomenal",
        "The speakers on the Lumina S8 Ultra are phenomenal — best I have heard on any phone. "
        "Bass is deep and treble is crisp. Great for music and videos. The display enhances "
        "the experience beautifully. Camera is sharp and accurate. Battery is excellent. "
        "Performance is incredibly smooth. Build quality feels solid and premium. "
        "Delivery was fast and packaging was excellent.", 5, True, 88),

    RawReview("R013","P002","Nisha K.","Camera is a game changer",
        "The camera on this phone is a complete game changer. Portrait mode is absolutely "
        "perfect, zoom quality is exceptional, and night shots are incredibly clear. "
        "Display is stunning with accurate colors. Performance handles heavy apps smoothly. "
        "Battery lasts more than a day easily. Build is solid and premium. "
        "Wifi connectivity is reliable and fast. Very happy with this purchase.", 5, True, 267),

    RawReview("R014","P002","Harish D.","Minor software issues otherwise perfect",
        "Everything about this phone is great except some minor software bugs. "
        "The performance is excellent, camera is outstanding, display is vibrant. "
        "Battery life is impressive. Build quality is premium. However I noticed a few "
        "bugs in the UI — sometimes notifications are delayed and the settings app crashes. "
        "Hoping a software update will fix these issues. Despite this, excellent phone.", 4, True, 134),

    RawReview("R015","P002","Divya M.","Connectivity issues disappoint",
        "The phone hardware is impressive — camera is great, display is stunning, "
        "performance is fast. However the wifi connectivity is very poor in my experience. "
        "Signal drops frequently and bluetooth range is disappointingly short. "
        "For a premium phone this is unacceptable. Battery and build quality are good. "
        "Camera and display make up for the connectivity issues but still disappointing.", 3, True, 119),

    # ─────────────────────────────────────────────
    # P003: SwiftPhone 7  (budget phone — value focused, known weaknesses)
    # ─────────────────────────────────────────────
    RawReview("R016","P003","Kartik B.","Best budget phone available",
        "For the price this phone is absolutely brilliant. Build quality is decent for a budget phone. "
        "Performance is smooth for everyday tasks — calls, social media, YouTube. "
        "Battery life is excellent — easily lasts two days on a single charge. "
        "Camera is decent but not outstanding — acceptable for casual photos. "
        "Display is bright enough outdoors. Value for money is great.", 5, True, 445),

    RawReview("R017","P003","Lalitha S.","Battery life is unmatched",
        "The battery life on this phone is incredible — I get nearly 3 days of moderate use. "
        "Fast charging is also good. Build quality is solid for a budget device. "
        "Performance handles daily tasks smoothly. Camera is not great for low light "
        "but daytime photos are decent. Display could be brighter. "
        "Overall very good value for money.", 4, True, 312),

    RawReview("R018","P003","Mohan R.","Camera is the weak point",
        "I bought this for my elderly parent who just needs a reliable phone. "
        "Performance is good, battery lasts long, build is sturdy. "
        "But the camera quality is poor — photos are blurry and grainy, especially indoors. "
        "The display is dim and dull compared to competitors. "
        "Audio from speakers is tinny and weak. For basic use it works fine.", 3, True, 87),

    RawReview("R019","P003","Sunita V.","Reliable daily driver",
        "This phone does everything I need reliably. Battery is excellent and "
        "charges fast. Performance is smooth — no lag in daily use. Build feels solid. "
        "Software is clean and simple with no bloatware. Camera takes good photos "
        "in daylight but struggles at night. For the price this is outstanding value. "
        "Delivery was quick and packaging was good.", 5, True, 193),

    RawReview("R020","P003","Prakash N.","Good for the price but display is poor",
        "The battery life is really impressive for such a cheap phone. "
        "Performance is acceptable. Build quality is decent. "
        "However the display resolution is low and colors look dull. "
        "Not suitable for watching videos. Camera also disappoints with grainy images. "
        "Audio quality from speakers is weak. Good for basic calling and messaging only.", 3, False, 64),

    RawReview("R021","P003","Rekha A.","Surprised by the performance",
        "I expected poor performance from a budget phone but this handles "
        "multitasking smoothly. Battery is fantastic — best in class for the price. "
        "Build quality is surprisingly solid. Camera is average but acceptable. "
        "Display could be better but is functional. Software is stable. "
        "Highly recommend for first time smartphone users.", 4, True, 156),

    RawReview("R022","P003","Vijay C.","Not worth even the budget price",
        "Performance lags badly with multiple apps open. Camera quality is terrible — "
        "blurry and grainy in all conditions. Display is dim and dull. "
        "Build quality feels very cheap and flimsy. Only the battery is decent. "
        "Software has annoying bugs. Not worth the money. Very disappointed.", 1, True, 231),
]
