from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import cv2
import numpy as np
import json
import uuid
from pathlib import Path

# ============================================================
# STYLE SENSE AI APPLICATION
# ============================================================

app = FastAPI(
    title="StyleSense AI",
    description="AI-Based Personalized Fashion Recommendation System",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# UPLOAD FOLDER
# ============================================================

UPLOAD_FOLDER = Path("uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads"
)


# ============================================================
# WARDROBE FOLDER
# ============================================================

WARDROBE_FOLDER = Path("wardrobe")
WARDROBE_FOLDER.mkdir(exist_ok=True)

WARDROBE_IMAGE_FOLDER = WARDROBE_FOLDER / "images"
WARDROBE_IMAGE_FOLDER.mkdir(exist_ok=True)

WARDROBE_FILE = WARDROBE_FOLDER / "wardrobe.json"

# ============================================================
# APPEARANCE PROFILE STORAGE
# ============================================================

APPEARANCE_FILE = Path("appearance.json")
PREFERENCES_FILE = Path("preferences.json")
# ============================================================
# RECOMMENDATION INDEX
# ============================================================

RECOMMENDATION_INDEX = 0
# ============================================================
# RECOMMENDATION HISTORY
# ============================================================

RECOMMENDATION_HISTORY = []


def outfit_signature(outfit):
    """
    Creates a unique signature for an outfit
    using the wardrobe item IDs.
    """

    ids = []

    for key in [
        "one_piece",
        "top",
        "bottom",
        "footwear",
        "accessory"
    ]:

        item = outfit.get(key)

        if item and item.get("id"):
            ids.append(item.get("id"))

    return tuple(sorted(ids))

# ============================================================
# SAVED OUTFITS STORAGE
# ============================================================

SAVED_OUTFITS_FOLDER = Path("saved_outfits")
SAVED_OUTFITS_FOLDER.mkdir(exist_ok=True)

SAVED_OUTFITS_FILE = (
    SAVED_OUTFITS_FOLDER / "saved_outfits.json"
)


if not SAVED_OUTFITS_FILE.exists():

    with open(
        SAVED_OUTFITS_FILE,
        "w"
    ) as file:

        json.dump(
            [],
            file,
            indent=4
        )


# ============================================================
# CREATE WARDROBE FILE IF NOT EXISTS
# ============================================================

if not WARDROBE_FILE.exists():

    with open(
        WARDROBE_FILE,
        "w"
    ) as file:

        json.dump(
            [],
            file,
            indent=4
        )


# ============================================================
# MAKE WARDROBE IMAGES ACCESSIBLE
# ============================================================

app.mount(
    "/wardrobe-images",
    StaticFiles(
        directory=str(WARDROBE_IMAGE_FOLDER)
    ),
    name="wardrobe-images"
)


# ============================================================
# PROFILE DATA STRUCTURE
# ============================================================

class Profile(BaseModel):

    name: str
    age: int
    gender: str
    style: str
    colour: str
    budget: str


# ============================================================
# SAVED OUTFIT DATA STRUCTURE
# ============================================================

class SavedOutfitRequest(BaseModel):

    outfit: dict
    preferences: dict = {}
    score: int = 0
    explanation: list[str] = []


# ============================================================
# STYLE PREFERENCES DATA STRUCTURE
# ============================================================

class Preferences(BaseModel):

    occasion: str
    weather: str
    style: str
    colour: str
    accessories: str
    budget: str


# ============================================================
# HOME API
# ============================================================

@app.get("/")
def home():
    frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    return FileResponse(frontend_path)

# ============================================================
# HEALTH CHECK API
# ============================================================

@app.get("/health")
def health_check():

    return {

        "status":
            "healthy"

    }


# ============================================================
# PROFILE API
# ============================================================

@app.post("/profile")
def create_profile(
    profile: Profile
):

    return {

        "message":
            "Profile received successfully",

        "profile":
            profile.model_dump()

    }


# ============================================================
# PHOTO UPLOAD API
# ============================================================

@app.post("/upload-photo")
async def upload_photo(
    file: UploadFile = File(...)
):

    if (
        not file.content_type
        or
        not file.content_type.startswith("image/")
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )


    file_path = (
        UPLOAD_FOLDER /
        file.filename
    )


    with open(
        file_path,
        "wb"
    ) as buffer:

        content = await file.read()

        buffer.write(content)


    return {

        "message":
            "Photo uploaded successfully",

        "filename":
            file.filename,

        "path":
            str(file_path)

    }

# ============================================================
# APPEARANCE ANALYSIS HELPER FUNCTIONS
# ============================================================

def estimate_skin_tone(face_image):
    """
    Estimate a broad skin-tone category from the face region.
    This is an approximate computer-vision estimate.
    """

    if face_image is None or face_image.size == 0:
        return "Unable to estimate"

    height, width = face_image.shape[:2]

    # Use the central/lower portion of the face
    # to reduce the influence of hair and background.
    y1 = int(height * 0.35)
    y2 = int(height * 0.85)

    x1 = int(width * 0.20)
    x2 = int(width * 0.80)

    skin_region = face_image[y1:y2, x1:x2]

    if skin_region.size == 0:
        return "Unable to estimate"

    # Convert BGR to HSV
    hsv = cv2.cvtColor(
        skin_region,
        cv2.COLOR_BGR2HSV
    )

    # Basic skin-colour filtering
    lower_skin = np.array(
        [0, 20, 50],
        dtype=np.uint8
    )

    upper_skin = np.array(
        [35, 255, 255],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower_skin,
        upper_skin
    )

    pixels = skin_region[mask > 0]

    if len(pixels) < 50:
        # Fallback: use the central region
        pixels = skin_region.reshape(
            -1,
            3
        )

    if len(pixels) == 0:
        return "Unable to estimate"

    # Calculate average brightness
    gray_values = cv2.cvtColor(
        pixels.reshape(-1, 1, 3),
        cv2.COLOR_BGR2GRAY
    )

    brightness = float(
        np.mean(gray_values)
    )

    # Broad categories only
    if brightness < 75:
        return "Deep"

    elif brightness < 115:
        return "Medium-Deep"

    elif brightness < 155:
        return "Medium"

    elif brightness < 195:
        return "Light-Medium"

    else:
        return "Light"


def estimate_hair_colour(image, face):

    """
    Estimate broad hair colour from the region
    above the detected face.

    This is an approximate computer-vision estimate.
    """

    x, y, width, height = face

    image_height, image_width = image.shape[:2]

    # Region above the face where hair may appear
    hair_y1 = max(
        0,
        y - int(height * 0.45)
    )

    hair_y2 = max(
        hair_y1 + 1,
        y + int(height * 0.10)
    )

    hair_x1 = max(
        0,
        x + int(width * 0.10)
    )

    hair_x2 = min(
        image_width,
        x + int(width * 0.90)
    )

    hair_region = image[
        hair_y1:hair_y2,
        hair_x1:hair_x2
    ]

    if hair_region.size == 0:
        return "Unable to estimate"

    hsv = cv2.cvtColor(
        hair_region,
        cv2.COLOR_BGR2HSV
    )

    mean_hue = float(
        np.mean(hsv[:, :, 0])
    )

    mean_saturation = float(
        np.mean(hsv[:, :, 1])
    )

    mean_value = float(
        np.mean(hsv[:, :, 2])
    )

    # Broad hair-colour classification
    if mean_value < 65:
        return "Black / Very Dark"

    elif mean_value < 100:

        if mean_saturation < 70:
            return "Dark Brown / Black"

        return "Dark Brown"

    elif mean_value < 145:

        if mean_hue < 20:
            return "Brown"

        return "Medium Brown"

    else:

        if mean_saturation < 60:
            return "Light / Grey"

        return "Light Brown"


def estimate_face_shape(face):

    """
    Estimate a basic face shape using the
    detected face bounding-box proportions.

    This is a simple geometric estimate.
    """

    x, y, width, height = face

    if width <= 0 or height <= 0:
        return "Unable to estimate"

    ratio = height / width

    if ratio >= 1.45:
        return "Oval / Long"

    elif ratio >= 1.25:
        return "Oval"

    elif ratio <= 1.05:
        return "Round / Broad"

    else:
        return "Balanced"


def generate_appearance_summary(
    face_detected,
    skin_tone,
    hair_colour,
    face_shape
):

    if not face_detected:

        return (
            "A clear face could not be detected. "
            "Please upload a clear front-facing photo."
        )

    return (
        "Face detected successfully. "
        f"Estimated skin tone: {skin_tone}. "
        f"Estimated hair colour: {hair_colour}. "
        f"Estimated face shape: {face_shape}. "
        "These appearance characteristics can be "
        "used as supporting information for "
        "personalized fashion recommendations."
    )

# ============================================================
# APPEARANCE ANALYSIS API
# ============================================================

@app.get("/analyze/{filename}")
def analyze_photo(
    filename: str
):

    file_path = (
        UPLOAD_FOLDER /
        filename
    )

    if not file_path.exists():

        raise HTTPException(
            status_code=404,
            detail="Photo not found."
        )

    image = cv2.imread(
        str(file_path)
    )

    if image is None:

        raise HTTPException(
            status_code=400,
            detail="Unable to read the image."
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    face_cascade = cv2.CascadeClassifier(

        cv2.data.haarcascades
        +
        "haarcascade_frontalface_default.xml"

    )

    faces = face_cascade.detectMultiScale(

        gray,

        scaleFactor=1.1,

        minNeighbors=5,

        minSize=(60, 60)

    )

    face_count = len(faces)

    # ========================================================
    # NO FACE DETECTED
    # ========================================================

    if face_count == 0:
        appearance_data = {

            "face_detected":
                False,

            "face_count":
                0,

            "skin_tone":
                "Unable to estimate",

            "hair_colour":
                "Unable to estimate",

            "face_shape":
                "Unable to estimate",

            "overall_appearance":
                generate_appearance_summary(
                    False,
                    "Unable to estimate",
                    "Unable to estimate",
                    "Unable to estimate"
                )

        }


        with open(
            APPEARANCE_FILE,
            "w"
        ) as file:

            json.dump(
                appearance_data,
                file,
                indent=4
            )

        return {

            "status":
                "success",

            "face_detected":
                False,

            "face_count":
                0,

            "skin_tone":
                "Unable to estimate",

            "hair_colour":
                "Unable to estimate",

            "face_shape":
                "Unable to estimate",

            "overall_appearance":
                generate_appearance_summary(
                    False,
                    "Unable to estimate",
                    "Unable to estimate",
                    "Unable to estimate"
                ),

            "message":
                "No clear face detected. "
                "Please try a front-facing photo."

        }

    # ========================================================
    # SELECT LARGEST FACE
    # ========================================================

    largest_face = max(

        faces,

        key=lambda face:
            face[2] * face[3]

    )

    x, y, width, height = (
        largest_face
    )

    # ========================================================
    # CROP FACE
    # ========================================================

    image_height, image_width = (
        image.shape[:2]
    )

    x1 = max(0, x)

    y1 = max(0, y)

    x2 = min(
        image_width,
        x + width
    )

    y2 = min(
        image_height,
        y + height
    )

    face_image = image[
        y1:y2,
        x1:x2
    ]

    # ========================================================
    # APPEARANCE ANALYSIS
    # ========================================================

    skin_tone = estimate_skin_tone(
        face_image
    )

    hair_colour = estimate_hair_colour(
        image,
        largest_face
    )

    face_shape = estimate_face_shape(
        largest_face
    )

    overall_appearance = (
        generate_appearance_summary(
            True,
            skin_tone,
            hair_colour,
            face_shape
        )
    )

    # ========================================================
    # SAVE APPEARANCE PROFILE
    # ========================================================

    appearance_data = {
        "face_detected": True,
        "face_count": face_count,
        "skin_tone": skin_tone,
        "hair_colour": hair_colour,
        "face_shape": face_shape,
        "overall_appearance": overall_appearance
    }

    with open(
        APPEARANCE_FILE,
        "w"
    ) as file:
        json.dump(
            appearance_data,
            file,
            indent=4
        )

    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {
        "status": "success",
        "face_detected": True,
        "face_count": face_count,
        "face_width": int(width),
        "face_height": int(height),
        "skin_tone": skin_tone,
        "hair_colour": hair_colour,
        "face_shape": face_shape,
        "overall_appearance": overall_appearance,
        "message": "Appearance analysis completed successfully."
    }
# ============================================================
# WARDROBE - ADD ITEM
# ============================================================

@app.post("/wardrobe-item")
async def add_wardrobe_item(

    item_name: str = Form(...),

    category: str = Form(...),

    colour: str = Form(...),

    pattern: str = Form(""),

    style: str = Form(""),

    occasion: str = Form(""),

    season: str = Form(""),

    file: UploadFile = File(...)

):

    if (
        not file.content_type
        or
        not file.content_type.startswith("image/")
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid clothing image."
        )


    item_id = str(
        uuid.uuid4()
    )


    original_name = (
        file.filename
        or
        "item.jpg"
    )

    extension = Path(
        original_name
    ).suffix.lower()


    if not extension:

        extension = ".jpg"


    image_filename = (
        item_id +
        extension
    )


    image_path = (
        WARDROBE_IMAGE_FOLDER /
        image_filename
    )


    with open(
        image_path,
        "wb"
    ) as buffer:

        content = await file.read()

        buffer.write(content)


    with open(
        WARDROBE_FILE,
        "r"
    ) as file:

        wardrobe = json.load(file)


    new_item = {

        "id":
            item_id,

        "item_name":
            item_name.strip(),

        "category":
            category.strip(),

        "colour":
            colour.strip(),

        "pattern":
            pattern.strip(),

        "style":
            style.strip(),

        "occasion":
            occasion.strip(),

        "season":
            season.strip(),

        "image":
            image_filename

    }


    wardrobe.append(
        new_item
    )


    with open(
        WARDROBE_FILE,
        "w"
    ) as file:

        json.dump(
            wardrobe,
            file,
            indent=4
        )


    return {

        "status":
            "success",

        "message":
            "Wardrobe item added successfully",

        "item":
            new_item

    }


# ============================================================
# WARDROBE - GET ALL ITEMS
# ============================================================

@app.get("/wardrobe")
def get_wardrobe():

    with open(
        WARDROBE_FILE,
        "r"
    ) as file:

        wardrobe = json.load(file)


    return {

        "status":
            "success",

        "count":
            len(wardrobe),

        "items":
            wardrobe

    }


# ============================================================
# WARDROBE - DELETE ITEM
# ============================================================

@app.delete("/wardrobe-item/{item_id}")
def delete_wardrobe_item(
    item_id: str
):

    with open(
        WARDROBE_FILE,
        "r"
    ) as file:

        wardrobe = json.load(file)


    item_to_delete = None


    for item in wardrobe:

        if item.get("id") == item_id:

            item_to_delete = item

            break


    if item_to_delete is None:

        raise HTTPException(
            status_code=404,
            detail="Wardrobe item not found."
        )


    wardrobe.remove(
        item_to_delete
    )


    with open(
        WARDROBE_FILE,
        "w"
    ) as file:

        json.dump(
            wardrobe,
            file,
            indent=4
        )


    return {

        "status":
            "success",

        "message":
            "Wardrobe item deleted successfully.",

        "item_id":
            item_id

    }


# ============================================================
# WARDROBE - EDIT ITEM
# ============================================================

@app.put("/wardrobe-item/{item_id}")
def edit_wardrobe_item(

    item_id: str,

    item_name: str = Form(...),

    category: str = Form(...),

    colour: str = Form(...),

    pattern: str = Form(""),

    style: str = Form(""),

    occasion: str = Form(""),

    season: str = Form("")

):

    with open(
        WARDROBE_FILE,
        "r"
    ) as file:

        wardrobe = json.load(file)


    item_to_edit = None


    for item in wardrobe:

        if item.get("id") == item_id:

            item_to_edit = item

            break


    if item_to_edit is None:

        raise HTTPException(
            status_code=404,
            detail="Wardrobe item not found."
        )


    item_to_edit["item_name"] = (
        item_name.strip()
    )

    item_to_edit["category"] = (
        category.strip()
    )

    item_to_edit["colour"] = (
        colour.strip()
    )

    item_to_edit["pattern"] = (
        pattern.strip()
    )

    item_to_edit["style"] = (
        style.strip()
    )

    item_to_edit["occasion"] = (
        occasion.strip()
    )

    item_to_edit["season"] = (
        season.strip()
    )


    with open(
        WARDROBE_FILE,
        "w"
    ) as file:

        json.dump(
            wardrobe,
            file,
            indent=4
        )


    return {

        "status":
            "success",

        "message":
            "Wardrobe item updated successfully.",

        "item":
            item_to_edit

    }


# ============================================================
# SAVE STYLE PREFERENCES
# ============================================================

@app.post("/preferences")
def save_preferences(
    preferences: Preferences
):

    preferences_file = Path(
        "preferences.json"
    )


    with open(
        preferences_file,
        "w"
    ) as file:

        json.dump(

            preferences.model_dump(),

            file,

            indent=4

        )


    return {

        "message":
            "Style preferences saved successfully",

        "preferences":
            preferences.model_dump()

    }


# ============================================================
# GET STYLE PREFERENCES
# ============================================================

@app.get("/preferences")
def get_preferences():

    preferences_file = Path(
        "preferences.json"
    )


    if not preferences_file.exists():

        return {

            "status":
                "empty",

            "message":
                "No preferences saved yet."

        }


    with open(
        preferences_file,
        "r"
    ) as file:

        preferences = json.load(file)


    return {

        "status":
            "success",

        "preferences":
            preferences

    }


# ============================================================
# SAVED OUTFITS - SAVE
# ============================================================

@app.post("/saved-outfit")
def save_outfit(
    request: SavedOutfitRequest
):

    if not request.outfit:

        raise HTTPException(

            status_code=400,

            detail="No outfit available to save."

        )


    with open(
        SAVED_OUTFITS_FILE,
        "r"
    ) as file:

        saved_outfits = json.load(file)


    if not isinstance(
        saved_outfits,
        list
    ):

        saved_outfits = []


    new_item_ids = sorted([

        item.get("id")

        for item in request.outfit.values()

        if item
        and
        item.get("id")

    ])


    for saved in saved_outfits:

        existing_outfit = saved.get(
            "outfit",
            {}
        )


        existing_item_ids = sorted([

            item.get("id")

            for item in existing_outfit.values()

            if item
            and
            item.get("id")

        ])


        if (
            new_item_ids
            ==
            existing_item_ids
        ):

            return {

                "status":
                    "duplicate",

                "message":
                    "This outfit is already saved."

            }


    saved_outfit = {

        "id":
            str(uuid.uuid4()),

        "outfit":
            request.outfit,

        "preferences":
            request.preferences,

        "score":
            request.score,

        "explanation":
            request.explanation

    }


    saved_outfits.append(
        saved_outfit
    )


    with open(
        SAVED_OUTFITS_FILE,
        "w"
    ) as file:

        json.dump(

            saved_outfits,

            file,

            indent=4

        )


    return {

        "status":
            "success",

        "message":
            "Outfit saved successfully.",

        "saved_outfit":
            saved_outfit

    }


# ============================================================
# SAVED OUTFITS - GET ALL
# ============================================================

@app.get("/saved-outfits")
def get_saved_outfits():

    with open(
        SAVED_OUTFITS_FILE,
        "r"
    ) as file:

        saved_outfits = json.load(file)


    if not isinstance(
        saved_outfits,
        list
    ):

        saved_outfits = []


    return {

        "status":
            "success",

        "count":
            len(saved_outfits),

        "outfits":
            saved_outfits

    }


# ============================================================
# SAVED OUTFITS - DELETE
# ============================================================

@app.delete("/saved-outfit/{outfit_id}")
def delete_saved_outfit(
    outfit_id: str
):

    with open(
        SAVED_OUTFITS_FILE,
        "r"
    ) as file:

        saved_outfits = json.load(file)


    outfit_to_delete = None


    for outfit in saved_outfits:

        if outfit.get("id") == outfit_id:

            outfit_to_delete = outfit

            break


    if outfit_to_delete is None:

        raise HTTPException(

            status_code=404,

            detail="Saved outfit not found."

        )


    saved_outfits.remove(
        outfit_to_delete
    )


    with open(
        SAVED_OUTFITS_FILE,
        "w"
    ) as file:

        json.dump(

            saved_outfits,

            file,

            indent=4

        )


    return {

        "status":
            "success",

        "message":
            "Saved outfit deleted successfully.",

        "outfit_id":
            outfit_id

    }


# ============================================================
# LOAD WARDROBE
# ============================================================

def load_wardrobe_data():

    with open(
        WARDROBE_FILE,
        "r"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD USER PREFERENCES
# ============================================================

def load_user_preferences():

    if not PREFERENCES_FILE.exists():
        return {}

    with open(
        PREFERENCES_FILE,
        "r"
    ) as file:
        return json.load(file)


# ============================================================
# LOAD APPEARANCE PROFILE
# ============================================================

def load_appearance_profile():

    if not APPEARANCE_FILE.exists():
        return {}

    with open(
        APPEARANCE_FILE,
        "r"
    ) as file:
        return json.load(file)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    value
):

    if value is None:

        return ""


    return str(
        value
    ).strip().lower()


# ============================================================
# WEATHER MAPPING
# ============================================================

def weather_to_season(
    weather
):

    weather = normalize_text(
        weather
    )


    mapping = {

        "hot": [
            "summer",
            "all season"
        ],

        "very hot": [
            "summer",
            "all season"
        ],

        "sunny": [
            "summer",
            "spring",
            "all season"
        ],

        "summer": [
            "summer",
            "spring",
            "all season"
        ],

        "rainy": [
            "monsoon",
            "all season"
        ],

        "rain": [
            "monsoon",
            "all season"
        ],

        "monsoon": [
            "monsoon",
            "all season"
        ],

        "cold": [
            "winter",
            "autumn",
            "all season"
        ],

        "very cold": [
            "winter",
            "all season"
        ],

        "winter": [
            "winter",
            "autumn",
            "all season"
        ],

        "pleasant": [
            "spring",
            "summer",
            "autumn",
            "all season"
        ],

        "mild": [
            "spring",
            "summer",
            "autumn",
            "all season"
        ]

    }


    return mapping.get(
        weather,
        []
    )


# ============================================================
# CATEGORY HELPERS
# ============================================================

def category_is(
    item,
    names
):

    category = normalize_text(
        item.get("category")
    )


    return category in [

        normalize_text(name)

        for name in names

    ]


# ============================================================
# ITEM NAME / CATEGORY SEARCH
# ============================================================

def item_matches_terms(
    item,
    terms
):

    category = normalize_text(
        item.get("category")
    )

    item_name = normalize_text(
        item.get("item_name")
    )


    for term in terms:

        term = normalize_text(
            term
        )


        if (
            category == term
            or
            term in item_name
            or
            term in category
        ):

            return True


    return False


# ============================================================
# TOP
# ============================================================

def is_top(
    item
):

    return category_is(

        item,

        [
            "top",
            "tops"
        ]

    )


# ============================================================
# BOTTOM
# ============================================================

def is_bottom(
    item
):

    return category_is(

        item,

        [
            "bottom",
            "bottoms",
            "pants",
            "trousers",
            "jeans",
            "skirt",
            "shorts"
        ]

    ) or item_matches_terms(

        item,

        [
            "pants",
            "trouser",
            "jeans",
            "skirt",
            "shorts"
        ]

    )


# ============================================================
# ONE-PIECE / COMPLETE OUTFIT
# ============================================================

def is_one_piece(
    item
):

    return item_matches_terms(

        item,

        [
            "dress",
            "dresses",
            "saree",
            "sari",
            "lehenga",
            "kurti",
            "kurta",
            "gown",
            "jumpsuit",
            "romper",
            "one piece",
            "one-piece",
            "anarkali"
        ]

    )


# ============================================================
# DRESS
# ============================================================

def is_dress(
    item
):

    return is_one_piece(
        item
    )


# ============================================================
# FOOTWEAR
# ============================================================

def is_footwear(
    item
):

    return category_is(

        item,

        [
            "footwear",
            "shoe",
            "shoes",
            "heels",
            "sneakers",
            "sandals"
        ]

    ) or item_matches_terms(

        item,

        [
            "shoe",
            "heels",
            "sneaker",
            "sandals",
            "slipper",
            "boots"
        ]

    )


# ============================================================
# ACCESSORY
# ============================================================

def is_accessory(
    item
):

    return category_is(

        item,

        [
            "accessory",
            "accessories",
            "jewellery",
            "jewelry",
            "bag",
            "handbag",
            "sunglasses",
            "watch"
        ]

    ) or item_matches_terms(

        item,

        [
            "jewellery",
            "jewelry",
            "necklace",
            "earring",
            "bracelet",
            "bag",
            "handbag",
            "purse",
            "sunglasses",
            "watch"
        ]

    )


# ============================================================
# ACCESSORY TYPE
# ============================================================

def accessory_type(
    item
):

    category = normalize_text(
        item.get("category")
    )

    name = normalize_text(
        item.get("item_name")
    )


    if (
        category in [
            "jewellery",
            "jewelry"
        ]
        or
        any(
            word in name
            for word in [
                "jewellery",
                "jewelry",
                "necklace",
                "earring",
                "bracelet",
                "ring"
            ]
        )
    ):

        return "jewellery"


    if (
        category in [
            "bag",
            "handbag"
        ]
        or
        any(
            word in name
            for word in [
                "bag",
                "handbag",
                "purse"
            ]
        )
    ):

        return "bag"


    if (
        category == "sunglasses"
        or
        "sunglasses" in name
    ):

        return "sunglasses"


    if (
        category in [
            "watch",
            "watches"
        ]
        or
        "watch" in name
    ):

        return "watch"


    return "accessory"


# ============================================================
# STYLE COMPATIBILITY GROUPS
# ============================================================

STYLE_GROUPS = {

    "casual": [
        "casual",
        "western",
        "minimal",
        "sporty"
    ],

    "formal": [
        "formal",
        "minimal"
    ],

    "party": [
        "party",
        "western",
        "formal"
    ],

    "traditional": [
        "traditional",
        "ethnic"
    ],

    "ethnic": [
        "ethnic",
        "traditional"
    ],

    "western": [
        "western",
        "casual",
        "party",
        "minimal"
    ],

    "sporty": [
        "sporty",
        "casual"
    ],

    "minimal": [
        "minimal",
        "casual",
        "formal",
        "western"
    ],

    "indo western": [
        "indo western",
        "western",
        "ethnic",
        "traditional",
        "party",
        "casual"
    ]

}
# ============================================================
# OCCASION COMPATIBILITY GROUPS
# ============================================================

OCCASION_GROUPS = {

    "college": [
        "college",
        "casual outing",
        "travel"
    ],

    "casual outing": [
        "casual outing",
        "college",
        "travel"
    ],

    "travel": [
        "travel",
        "casual outing",
        "college"
    ],

    "office": [
        "office",
        "formal"
    ],

    "formal": [
        "formal",
        "office"
    ],

    "party": [
        "party",
        "date",
        "wedding",
        "reception"
    ],

    "date": [
        "date",
        "party",
        "casual outing"
    ],

    "wedding": [
        "wedding",
        "traditional event",
        "reception",
        "party"
    ],

    "reception": [
        "reception",
        "wedding",
        "party"
    ],

    "traditional event": [
        "traditional event",
        "wedding",
        "reception"
    ],

    "sports": [
        "sports"
    ]

}
# ============================================================
# OCCASION RELATIONSHIP
# ============================================================

def occasion_relationship(
    item_occasion,
    preferred_occasion
):

    item_occasion = normalize_text(
        item_occasion
    )

    preferred_occasion = normalize_text(
        preferred_occasion
    )

    if not preferred_occasion:
        return 0

    if not item_occasion:
        return 5

    if item_occasion == preferred_occasion:
        return 30

    compatible = OCCASION_GROUPS.get(
        preferred_occasion,
        []
    )

    if item_occasion in compatible:
        return 18

    return -25


# ============================================================
# FILTER ITEMS BY OCCASION
# ============================================================

def filter_items_by_occasion(

    items,

    preferred_occasion,

    fallback_to_all=True

):

    if not items or not preferred_occasion:
        return items

    compatible_items = [
        item
        for item in items
        if occasion_relationship(
            item.get("occasion"),
            preferred_occasion
        ) >= 18
    ]

    # Use compatible items whenever the wardrobe has them.
    # For optional categories, the caller can disable fallback so
    # an incompatible item is omitted instead of being recommended.
    if compatible_items:
        return compatible_items

    return items if fallback_to_all else []


# ============================================================
# STYLE RELATIONSHIP
# ============================================================

def style_relationship(
    item_style,
    preferred_style
):

    item_style = normalize_text(
        item_style
    )

    preferred_style = normalize_text(
        preferred_style
    )

    if not preferred_style:
        return 0

    if not item_style:
        return 5

    if item_style == preferred_style:
        return 30

    compatible = STYLE_GROUPS.get(
        preferred_style,
        []
    )

    if item_style in compatible:
        return 18

    return -18


# ============================================================
# SEASON / WEATHER RELATIONSHIP
# ============================================================

def season_relationship(
    item_season,
    preferred_weather
):

    item_season = normalize_text(
        item_season
    )

    preferred_weather = normalize_text(
        preferred_weather
    )

    if not preferred_weather:
        return 0

    if not item_season:
        return 5

    if item_season == "all season":
        return 20

    compatible_seasons = weather_to_season(
        preferred_weather
    )

    if item_season in compatible_seasons:
        return 20

    return -12
# ============================================================
# COLOUR COMPATIBILITY
# ============================================================

NEUTRAL_COLOURS = [

    "black",
    "white",
    "cream",
    "beige",
    "grey",
    "gray",
    "navy",
    "brown"

]


COMPATIBLE_COLOUR_PAIRS = {

    "red": [

        "black",
        "white",
        "cream",
        "beige",
        "navy",
        "grey",
        "gray",
        "pink"

    ],

    "blue": [

        "black",
        "white",
        "cream",
        "beige",
        "grey",
        "gray",
        "navy"

    ],

    "green": [

        "black",
        "white",
        "cream",
        "beige",
        "grey",
        "gray",
        "brown"

    ],

    "lavender": [

        "black",
        "white",
        "cream",
        "grey",
        "gray",
        "pink",
        "navy",
        "purple"

    ],

    "levender": [

        "black",
        "white",
        "cream",
        "grey",
        "gray",
        "pink",
        "navy",
        "purple"

    ],

    "purple": [

        "black",
        "white",
        "cream",
        "grey",
        "gray",
        "pink"

    ],

    "pink": [

        "black",
        "white",
        "grey",
        "gray",
        "navy",
        "red",
        "cream"

    ],

    "orange": [

        "black",
        "white",
        "cream",
        "beige",
        "brown"

    ],

    "yellow": [

        "black",
        "white",
        "navy",
        "grey",
        "gray",
        "brown"

    ],

    "brown": [

        "white",
        "cream",
        "beige",
        "black",
        "blue",
        "green"

    ]

}
# ============================================================
# SKIN-TONE COLOUR COMPATIBILITY
# ============================================================

SKIN_TONE_COLOURS = {

    "deep": [
        "white",
        "cream",
        "beige",
        "red",
        "pink",
        "purple",
        "orange",
        "yellow",
        "green",
        "blue",
        "navy"
    ],

    "medium-deep": [
        "cream",
        "beige",
        "red",
        "maroon",
        "pink",
        "purple",
        "green",
        "blue",
        "navy",
        "brown"
    ],

    "medium": [
        "cream",
        "beige",
        "red",
        "pink",
        "purple",
        "green",
        "blue",
        "navy",
        "brown",
        "white"
    ],

    "light-medium": [
        "white",
        "cream",
        "beige",
        "pink",
        "purple",
        "blue",
        "green",
        "red",
        "navy"
    ],

    "light": [
        "white",
        "cream",
        "pink",
        "red",
        "blue",
        "purple",
        "green",
        "navy",
        "brown"
    ]

}

# ============================================================
# COLOUR COMPATIBILITY SCORE
# ============================================================

def calculate_colour_compatibility(
    colour1,
    colour2
):

    colour1 = normalize_text(
        colour1
    )

    colour2 = normalize_text(
        colour2
    )


    if not colour1 or not colour2:

        return 0


    if colour1 == colour2:

        return 8


    if (
        colour1 in NEUTRAL_COLOURS
        or
        colour2 in NEUTRAL_COLOURS
    ):

        return 10


    if (
        colour2
        in
        COMPATIBLE_COLOUR_PAIRS.get(
            colour1,
            []
        )
    ):

        return 8


    if (
        colour1
        in
        COMPATIBLE_COLOUR_PAIRS.get(
            colour2,
            []
        )
    ):

        return 8


    return 2

# ============================================================
# ITEM PREFERENCE SCORE
# ============================================================

def calculate_item_score(

    item,

    preferences

):

    score = 0
    reasons = []

    item_colour = normalize_text(item.get("colour"))
    item_style = normalize_text(item.get("style"))
    item_occasion = normalize_text(item.get("occasion"))
    item_season = normalize_text(item.get("season"))
    item_pattern = normalize_text(item.get("pattern"))

    preferred_colour = normalize_text(preferences.get("colour"))
    preferred_style = normalize_text(preferences.get("style"))
    preferred_occasion = normalize_text(preferences.get("occasion"))
    preferred_weather = normalize_text(preferences.get("weather"))

    appearance = load_appearance_profile()
    skin_tone = normalize_text(appearance.get("skin_tone"))

    # Occasion
    occasion_score = occasion_relationship(
        item_occasion,
        preferred_occasion
    )
    score += occasion_score

    if occasion_score >= 30:
        reasons.append("occasion matched")
    elif occasion_score >= 18:
        reasons.append("compatible occasion")
    elif occasion_score < 0:
        reasons.append("occasion mismatch")
    elif not item_occasion:
        reasons.append("occasion not specified")

    # Style
    style_score = style_relationship(
        item_style,
        preferred_style
    )
    score += style_score

    if style_score >= 30:
        reasons.append("style matched")
    elif style_score >= 18:
        reasons.append("compatible style")
    elif style_score < 0:
        reasons.append("style mismatch")
    elif not item_style:
        reasons.append("style not specified")

    # Weather / season
    season_score = season_relationship(
        item_season,
        preferred_weather
    )
    score += season_score

    if season_score >= 20:
        reasons.append("weather matched")
    elif season_score < 0:
        reasons.append("weather mismatch")
    elif not item_season:
        reasons.append("season not specified")

    # Preferred colour
    if preferred_colour:
        if item_colour == preferred_colour:
            score += 15
            reasons.append("preferred colour matched")
        elif item_colour in NEUTRAL_COLOURS:
            score += 10
            reasons.append("neutral colour")
        elif item_colour in COMPATIBLE_COLOUR_PAIRS.get(
            preferred_colour,
            []
        ):
            score += 8
            reasons.append("compatible colour")
        elif not item_colour:
            score += 3
            reasons.append("colour not specified")
        else:
            score += 1
            reasons.append("different colour")

    # Appearance / skin tone
    if skin_tone and item_colour:
        suitable_colours = SKIN_TONE_COLOURS.get(
            skin_tone,
            []
        )

        if item_colour in suitable_colours:
            score += 6
            reasons.append(
                "colour is compatible with the estimated skin tone"
            )
        elif item_colour in NEUTRAL_COLOURS:
            score += 3
            reasons.append(
                "neutral colour considered suitable for the estimated skin tone"
            )

    # Pattern
    if item_pattern:
        score += 5
        reasons.append("pattern considered")
    else:
        score += 2

    score = max(
        0,
        min(round(score), 100)
    )

    return score, reasons


# ============================================================
# ITEM RESULT
# ============================================================

def score_item(

    item,

    preferences

):

    score, reasons = calculate_item_score(

        item,

        preferences

    )


    return {

        "item":
            item,

        "score":
            score,

        "reasons":
            reasons

    }

# ============================================================
# PAIR COMPATIBILITY
# ============================================================

def calculate_pair_compatibility(

    item1,

    item2

):

    if not item1 or not item2:
        return 0

    score = 0

    style1 = normalize_text(item1.get("style"))
    style2 = normalize_text(item2.get("style"))
    occasion1 = normalize_text(item1.get("occasion"))
    occasion2 = normalize_text(item2.get("occasion"))
    colour1 = normalize_text(item1.get("colour"))
    colour2 = normalize_text(item2.get("colour"))

    if style1 and style2:
        if style1 == style2:
            score += 10
        elif (
            style2 in STYLE_GROUPS.get(style1, [])
            or style1 in STYLE_GROUPS.get(style2, [])
        ):
            score += 7
        else:
            score -= 8

    if occasion1 and occasion2:
        if occasion1 == occasion2:
            score += 10
        elif (
            occasion2 in OCCASION_GROUPS.get(occasion1, [])
            or occasion1 in OCCASION_GROUPS.get(occasion2, [])
        ):
            score += 6
        else:
            score -= 8

    score += calculate_colour_compatibility(
        colour1,
        colour2
    )

    return max(
        -20,
        min(score, 30)
    )


# ============================================================
# COMPLETE OUTFIT SCORE
# ============================================================

def calculate_complete_outfit_score(

    results,

    preferences

):

    valid_results = [

        result

        for result in results

        if result is not None

    ]


    if not valid_results:

        return 0


    # ========================================================
    # ITEM SCORE
    # ========================================================

    item_total = sum(

        result["score"]

        for result in valid_results

    )


    item_average = (

        item_total
        /
        len(valid_results)

    )


    # ========================================================
    # PAIR COMPATIBILITY
    # ========================================================

    compatibility_scores = []


    for i in range(
        len(valid_results)
    ):

        for j in range(
            i + 1,
            len(valid_results)
        ):

            compatibility_scores.append(

                calculate_pair_compatibility(

                    valid_results[i]["item"],

                    valid_results[j]["item"]

                )

            )


    if compatibility_scores:

        compatibility_average = (

            sum(
                compatibility_scores
            )
            /
            len(
                compatibility_scores
            )

        )


        compatibility_percentage = (

            compatibility_average
            /
            30

        ) * 100

    else:

        compatibility_percentage = 0


    # ========================================================
    # FINAL SCORE
    # ========================================================

    final_score = (

        (
            item_average
            *
            0.70
        )

        +

        (
            compatibility_percentage
            *
            0.30
        )

    )


    return max(

        0,

        min(

            round(
                final_score
            ),

            100

        )

    )


# ============================================================
# FIND BEST OUTFIT
# ============================================================

def find_best_outfit(

    tops,
    bottoms,
    footwear,
    accessories,
    one_piece_items,
    preferences,
    outfit_index=0

):

    # Prefer wardrobe items suitable for the selected occasion.
    # For footwear/accessories we fall back when the wardrobe has no
    # occasion-compatible item because these categories may be sparse.
    preferred_occasion = preferences.get("occasion")

    occasion_tops = [
        item
        for item in tops
        if occasion_relationship(
            item.get("occasion"),
            preferred_occasion
        ) >= 18
    ]

    occasion_bottoms = [
        item
        for item in bottoms
        if occasion_relationship(
            item.get("occasion"),
            preferred_occasion
        ) >= 18
    ]

    occasion_one_piece = [
        item
        for item in one_piece_items
        if occasion_relationship(
            item.get("occasion"),
            preferred_occasion
        ) >= 18
    ]

    occasion_footwear = filter_items_by_occasion(
        footwear,
        preferred_occasion,
        fallback_to_all=False
    )

    occasion_accessories = filter_items_by_occasion(
        accessories,
        preferred_occasion,
        fallback_to_all=False
    )

    # If a complete occasion-compatible one-piece exists but there is
    # no compatible top+bottom pair, prefer the one-piece route.
    if occasion_one_piece and not (
        occasion_tops and occasion_bottoms
    ):
        tops = []
        bottoms = []
        one_piece_items = occasion_one_piece
    else:
        tops = occasion_tops if occasion_tops else tops
        bottoms = occasion_bottoms if occasion_bottoms else bottoms
        one_piece_items = (
            occasion_one_piece
            if occasion_one_piece
            else one_piece_items
        )

    footwear = occasion_footwear
    accessories = occasion_accessories

    combinations = []

    accessory_options = accessories if accessories else [None]
    footwear_options = footwear if footwear else [None]

    # One-piece outfits
    for one_piece in one_piece_items:
        one_piece_result = score_item(
            one_piece,
            preferences
        )

        for shoe in footwear_options:
            shoe_result = (
                score_item(shoe, preferences)
                if shoe else None
            )

            for accessory in accessory_options:
                accessory_result = (
                    score_item(accessory, preferences)
                    if accessory else None
                )

                results = [
                    one_piece_result,
                    shoe_result,
                    accessory_result
                ]

                combinations.append({
                    "one_piece": one_piece_result,
                    "footwear": shoe_result,
                    "accessory": accessory_result,
                    "top": None,
                    "bottom": None,
                    "score": calculate_complete_outfit_score(
                        results,
                        preferences
                    )
                })

    # Top + bottom outfits
    for top in tops:
        top_result = score_item(
            top,
            preferences
        )

        for bottom in bottoms:
            bottom_result = score_item(
                bottom,
                preferences
            )

            for shoe in footwear_options:
                shoe_result = (
                    score_item(shoe, preferences)
                    if shoe else None
                )

                for accessory in accessory_options:
                    accessory_result = (
                        score_item(accessory, preferences)
                        if accessory else None
                    )

                    results = [
                        top_result,
                        bottom_result,
                        shoe_result,
                        accessory_result
                    ]

                    combinations.append({
                        "top": top_result,
                        "bottom": bottom_result,
                        "footwear": shoe_result,
                        "accessory": accessory_result,
                        "one_piece": None,
                        "score": calculate_complete_outfit_score(
                            results,
                            preferences
                        )
                    })

    if not combinations:
        return None

    combinations.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Remove exact duplicate outfits.
    unique_combinations = []
    seen = set()

    for combination in combinations:
        ids = tuple(sorted([
            result["item"].get("id")
            for key in [
                "one_piece",
                "top",
                "bottom",
                "footwear",
                "accessory"
            ]
            if (
                result := combination.get(key)
            ) and result.get("item")
        ]))

        if ids not in seen:
            seen.add(ids)
            unique_combinations.append(combination)

    combinations = unique_combinations

    # Prefer combinations with a reasonable score, but never fail
    # solely because the wardrobe is small.
    quality_combinations = [
        combination
        for combination in combinations
        if combination.get("score", 0) >= 50
    ]

    if quality_combinations:
        combinations = quality_combinations
    else:
        combinations = combinations[:1]

    selected_index = (
        outfit_index % len(combinations)
    )

    return combinations[selected_index]


# ============================================================
# BUILD OUTFIT EXPLANATION
# ============================================================

def build_outfit_explanation(

    selected,

    preferences

):

    explanation = []


    score = selected[
        "score"
    ]


    explanation.append(

        f"Outfit compatibility score: "
        f"{score}/100."

    )

    # ========================================================
    # APPEARANCE INFORMATION
    # ========================================================

    appearance = load_appearance_profile()

    skin_tone = appearance.get(
        "skin_tone"
    )

    hair_colour = appearance.get(
        "hair_colour"
    )

    face_shape = appearance.get(
        "face_shape"
    )


    if skin_tone and skin_tone != "Unable to estimate":

        explanation.append(

            f"Estimated skin tone considered: "
            f"{skin_tone}."

        )


    if hair_colour and hair_colour != "Unable to estimate":

        explanation.append(

            f"Estimated hair colour available for "
            f"personalization: {hair_colour}."

        )


    if face_shape and face_shape != "Unable to estimate":

        explanation.append(

            f"Estimated face shape available for "
            f"personalization: {face_shape}."

        )
    # ========================================================
    # PREFERENCES
    # ========================================================

    if preferences.get(
        "occasion"
    ):

        explanation.append(

            f"Occasion considered: "
            f"{preferences['occasion']}."

        )


    if preferences.get(
        "style"
    ):

        explanation.append(

            f"Preferred style considered: "
            f"{preferences['style']}."

        )


    if preferences.get(
        "colour"
    ):

        explanation.append(

            f"Preferred colour considered: "
            f"{preferences['colour']}."

        )


    if preferences.get(
        "weather"
    ):

        explanation.append(

            f"Weather considered: "
            f"{preferences['weather']}."

        )


    if preferences.get(
        "accessories"
    ):

        explanation.append(

            f"Accessory preference considered: "
            f"{preferences['accessories']}."

        )


    # ========================================================
    # ITEMS
    # ========================================================

    categories = [

        "one_piece",
        "top",
        "bottom",
        "footwear",
        "accessory"

    ]


    for category in categories:

        result = selected.get(
            category
        )


        if not result:

            continue


        item = result[
            "item"
        ]


        reasons = result[
            "reasons"
        ]


        item_name = item.get(

            "item_name",

            "Item"

        )


        item_colour = item.get(

            "colour",

            ""

        )


        item_style = item.get(

            "style",

            ""

        )


        item_occasion = item.get(

            "occasion",

            ""

        )


        explanation.append(

            f"{item_name} was selected as "
            f"{category.replace('_', ' ')}."

        )


        simple_reasons = []


        # ----------------------------------------------------
        # OCCASION
        # ----------------------------------------------------

        if item_occasion:

            preferred_occasion = preferences.get(

                "occasion",

                ""

            )


            if (

                preferred_occasion

                and

                normalize_text(
                    item_occasion
                )
                ==
                normalize_text(
                    preferred_occasion
                )

            ):

                simple_reasons.append(

                    f"it matches your "
                    f"{preferred_occasion} occasion"

                )


        # ----------------------------------------------------
        # STYLE
        # ----------------------------------------------------

        if item_style:

            preferred_style = preferences.get(

                "style",

                ""

            )


            if (

                preferred_style

                and

                normalize_text(
                    item_style
                )
                ==
                normalize_text(
                    preferred_style
                )

            ):

                simple_reasons.append(

                    f"it matches your "
                    f"{preferred_style} style"

                )

            elif (

                preferred_style

                and

                normalize_text(
                    item_style
                )
                in
                STYLE_GROUPS.get(
                    normalize_text(
                        preferred_style
                    ),
                    []
                )

            ):

                simple_reasons.append(

                    f"its style is compatible "
                    f"with your preferred "
                    f"{preferred_style} style"

                )


        # ----------------------------------------------------
        # COLOUR
        # ----------------------------------------------------

        preferred_colour = preferences.get(

            "colour",

            ""

        )


        if (
            preferred_colour
            and
            item_colour
        ):

            if (

                normalize_text(
                    item_colour
                )
                ==
                normalize_text(
                    preferred_colour
                )

            ):

                simple_reasons.append(

                    f"it matches your preferred "
                    f"{preferred_colour} colour"

                )

            elif (

                normalize_text(
                    item_colour
                )
                in
                NEUTRAL_COLOURS

            ):

                simple_reasons.append(

                    f"{item_colour} provides a "
                    f"neutral alternative to your "
                    f"preferred {preferred_colour}"

                )

            elif (

                normalize_text(
                    item_colour
                )
                in
                COMPATIBLE_COLOUR_PAIRS.get(
                    normalize_text(
                        preferred_colour
                    ),
                    []
                )

            ):

                simple_reasons.append(

                    f"{item_colour} is compatible "
                    f"with your preferred "
                    f"{preferred_colour} colour"

                )


        # ----------------------------------------------------
        # ADD REASON
        # ----------------------------------------------------

        if simple_reasons:

            explanation.append(

                "Why it was selected: "
                +
                ", ".join(
                    simple_reasons
                )
                +
                "."

            )


        # ----------------------------------------------------
        # WEATHER
        # ----------------------------------------------------

        if (
            "weather matched"
            in
            reasons
        ):

            explanation.append(

                f"{item_name} is suitable for "
                f"the selected "
                f"{preferences.get('weather')} "
                f"weather."

            )


    # ========================================================
    # WARDROBE SOURCE
    # ========================================================

    explanation.append(

        "The recommendation uses items available "
        "in your personal wardrobe."

    )


    return explanation
# ============================================================
# COMPLETE MY LOOK
# ============================================================

def find_compatible_item(

    selected_item,

    candidate_items,

    preferences,

    excluded_ids=None

):

    if not candidate_items:

        return None


    # ========================================================
    # PREVIOUSLY USED ITEMS
    # ========================================================

    if excluded_ids is None:

        excluded_ids = set()

    else:

        excluded_ids = set(excluded_ids)


    scored_candidates = []


    # ========================================================
    # SCORE CANDIDATE ITEMS
    # ========================================================

    for item in candidate_items:

        # ----------------------------------------------------
        # SKIP PREVIOUSLY USED ITEM
        # ----------------------------------------------------

        if item.get("id") in excluded_ids:

            continue


        preference_score, reasons = calculate_item_score(

            item,

            preferences

        )


        compatibility_score = calculate_pair_compatibility(

            selected_item,

            item

        )


        final_score = (

            (

                preference_score
                *
                0.60

            )

            +

            (

                compatibility_score
                *
                1.33

            )

        )


        final_score = round(

            min(

                final_score,

                100

            )

        )


        item_reasons = []


        # ====================================================
        # COMPATIBILITY REASON
        # ====================================================

        if compatibility_score >= 20:

            item_reasons.append(

                "strong compatibility with "
                "the selected item"

            )

        elif compatibility_score >= 12:

            item_reasons.append(

                "good compatibility with "
                "the selected item"

            )

        else:

            item_reasons.append(

                "available wardrobe option"

            )


        # ====================================================
        # STYLE REASON
        # ====================================================

        item_style = normalize_text(

            item.get("style")

        )


        selected_style = normalize_text(

            selected_item.get("style")

        )


        if (

            item_style
            and
            selected_style
            and
            item_style == selected_style

        ):

            item_reasons.append(

                f"matches the {item_style} style"

            )


        # ====================================================
        # OCCASION REASON
        # ====================================================

        item_occasion = normalize_text(

            item.get("occasion")

        )


        preferred_occasion = normalize_text(

            preferences.get("occasion")

        )


        if (

            item_occasion
            and
            preferred_occasion
            and
            item_occasion == preferred_occasion

        ):

            item_reasons.append(

                f"suitable for "
                f"{preferences.get('occasion')}"

            )


        # ====================================================
        # COLOUR REASON
        # ====================================================

        item_colour = item.get(

            "colour",

            ""

        )


        if item_colour:

            item_reasons.append(

                f"uses {item_colour} colour"

            )


        # ====================================================
        # ADD CANDIDATE
        # ====================================================

        scored_candidates.append({

            "item":
                item,

            "score":
                final_score,

            "reasons":
                item_reasons

        })


    # ========================================================
    # NO AVAILABLE CANDIDATES
    # ========================================================

    if not scored_candidates:

        return None


    # ========================================================
    # SORT BY SCORE
    # ========================================================

    scored_candidates.sort(

        key=lambda x:
            x["score"],

        reverse=True

    )


    # ========================================================
    # RETURN BEST AVAILABLE ITEM
    # ========================================================

    return scored_candidates[0]

# ============================================================
# BUILD COMPLETE LOOK
# ============================================================

def build_complete_look(

    selected_item,

    wardrobe,

    preferences,

    excluded_items=None

):

    if excluded_items is None:

        excluded_items = {}


    complete_look = {

        "selected_item":
            selected_item,

        "items":
            {},

        "score":
            0,

        "explanation":
            []

    }


    selected_is_one_piece = is_one_piece(
        selected_item
    )


    selected_is_top = is_top(
        selected_item
    )


    selected_is_bottom = is_bottom(
        selected_item
    )


    # ========================================================
    # FIND BOTTOMS
    # ========================================================

    bottoms = [

        item

        for item in wardrobe

        if is_bottom(item)

        and
        item.get("id")
        !=
        selected_item.get("id")

    ]


    # ========================================================
    # FIND TOPS
    # ========================================================

    tops = [

        item

        for item in wardrobe

        if is_top(item)

        and
        item.get("id")
        !=
        selected_item.get("id")

    ]


    # ========================================================
    # FIND FOOTWEAR
    # ========================================================

    footwear = [

        item

        for item in wardrobe

        if is_footwear(item)

        and
        item.get("id")
        !=
        selected_item.get("id")

    ]


    # ========================================================
    # FIND ACCESSORIES
    # ========================================================

    accessories = [

        item

        for item in wardrobe

        if is_accessory(item)

        and
        item.get("id")
        !=
        selected_item.get("id")

    ]


    # ========================================================
    # SEPARATE ACCESSORIES
    # ========================================================

    jewellery = [

        item

        for item in accessories

        if accessory_type(item)
        ==
        "jewellery"

    ]


    bags = [

        item

        for item in accessories

        if accessory_type(item)
        ==
        "bag"

    ]


    sunglasses = [

        item

        for item in accessories

        if accessory_type(item)
        ==
        "sunglasses"

    ]


    watches = [

        item

        for item in accessories

        if accessory_type(item)
        ==
        "watch"

    ]


    other_accessories = [

        item

        for item in accessories

        if accessory_type(item)
        ==
        "accessory"

    ]


    # ========================================================
    # IF SELECTED ITEM IS A TOP
    # ========================================================

    if selected_is_top:

        bottom_result = find_compatible_item(

            selected_item,

            bottoms,

            preferences,

            excluded_items.get(
                "bottom",
                []
            )

        )


        if bottom_result:

            complete_look["items"]["bottom"] = (

                bottom_result["item"]

            )


    # ========================================================
    # IF SELECTED ITEM IS A BOTTOM
    # ========================================================

    elif selected_is_bottom:

        top_result = find_compatible_item(

            selected_item,

            tops,

            preferences,

            excluded_items.get(
                "top",
                []
            )

        )


        if top_result:

            complete_look["items"]["top"] = (

                top_result["item"]

            )


    # ========================================================
    # IF SELECTED ITEM IS ONE-PIECE
    # ========================================================

    elif selected_is_one_piece:

        # Dress / Saree / Lehenga / Kurti etc.
        # are already complete clothing items.

        pass


    # ========================================================
    # FOOTWEAR
    # ========================================================

    footwear_result = find_compatible_item(

        selected_item,

        footwear,

        preferences,

        excluded_items.get(
            "footwear",
            []
        )

    )


    if footwear_result:

        complete_look["items"]["footwear"] = (

            footwear_result["item"]

        )


    # ========================================================
    # BAG
    # ========================================================

    bag_result = find_compatible_item(

        selected_item,

        bags,

        preferences,

        excluded_items.get(
            "bag",
            []
        )

    )


    if bag_result:

        complete_look["items"]["bag"] = (

            bag_result["item"]

        )


    # ========================================================
    # JEWELLERY
    # ========================================================

    jewellery_result = find_compatible_item(

        selected_item,

        jewellery,

        preferences,

        excluded_items.get(
            "jewellery",
            []
        )

    )


    if jewellery_result:

        complete_look["items"]["jewellery"] = (

            jewellery_result["item"]

        )


    # ========================================================
    # SUNGLASSES
    # ========================================================

    sunglasses_result = find_compatible_item(

        selected_item,

        sunglasses,

        preferences,

        excluded_items.get(
            "sunglasses",
            []
        )

    )


    if sunglasses_result:

        complete_look["items"]["sunglasses"] = (

            sunglasses_result["item"]

        )


    # ========================================================
    # WATCH
    # ========================================================

    watch_result = find_compatible_item(

        selected_item,

        watches,

        preferences,

        excluded_items.get(
            "watch",
            []
        )

    )


    if watch_result:

        complete_look["items"]["watch"] = (

            watch_result["item"]

        )


    # ========================================================
    # OTHER ACCESSORY
    # ========================================================

    other_result = find_compatible_item(

        selected_item,

        other_accessories,

        preferences,

        excluded_items.get(
            "accessory",
            []
        )

    )


    if other_result:

        complete_look["items"]["accessory"] = (

            other_result["item"]

        )


    # ========================================================
    # OVERALL SCORE
    # ========================================================

    compatibility_scores = []


    for item in complete_look[
        "items"
    ].values():

        compatibility_scores.append(

            calculate_pair_compatibility(

                selected_item,

                item

            )

        )


    if compatibility_scores:

        average_compatibility = (

            sum(
                compatibility_scores
            )
            /
            len(
                compatibility_scores
            )

        )


        complete_look["score"] = round(

            min(

                (
                    average_compatibility
                    /
                    30
                )
                *
                100,

                100

            )

        )


    else:

        complete_look["score"] = 0


    # ========================================================
    # EXPLANATION
    # ========================================================

    selected_name = selected_item.get(

        "item_name",

        "selected item"

    )


    complete_look["explanation"].append(

        f"StyleSense AI used {selected_name} "
        f"as the main item for this look."

    )


    if selected_is_one_piece:

        complete_look["explanation"].append(

            f"{selected_name} is treated as a "
            "complete outfit item, so no separate "
            "bottom was added."

        )


    if complete_look["items"]:

        complete_look["explanation"].append(

            "Additional items were selected from "
            "your personal wardrobe based on colour, "
            "style and occasion compatibility."

        )


    if preferences.get("occasion"):

        complete_look["explanation"].append(

            f"The selected occasion was "
            f"{preferences.get('occasion')}."

        )


    if preferences.get("style"):

        complete_look["explanation"].append(

            f"Your preferred style was "
            f"{preferences.get('style')}."

        )


    if preferences.get("weather"):

        complete_look["explanation"].append(

            f"The selected weather was "
            f"{preferences.get('weather')}."

        )


    for category, item in complete_look[
        "items"
    ].items():

        complete_look["explanation"].append(

            f"{item.get('item_name', 'Item')} "
            f"was selected as {category} because it "
            f"complements the selected "
            f"{selected_name}."

        )


    return complete_look


# ============================================================
# COMPLETE MY LOOK API
# ============================================================

@app.get("/complete-my-look/{item_id}")
def complete_my_look(

    item_id: str

):

    wardrobe = load_wardrobe_data()

    preferences = load_user_preferences()


    if not wardrobe:

        raise HTTPException(

            status_code=400,

            detail="Your wardrobe is empty."

        )


    selected_item = None


    for item in wardrobe:

        if item.get("id") == item_id:

            selected_item = item

            break


    if selected_item is None:

        raise HTTPException(

            status_code=404,

            detail="Selected wardrobe item not found."

        )


    complete_look = build_complete_look(

        selected_item,

        wardrobe,

        preferences

    )


    return {

        "status":
            "success",

        "message":
            "Complete look generated successfully.",

        "selected_item":
            complete_look["selected_item"],

        "items":
            complete_look["items"],

        "score":
            complete_look["score"],

        "preferences":
            preferences,

        "explanation":
            complete_look["explanation"]

    }


# ============================================================
# GENERATE ANOTHER COMPLETE LOOK
# ============================================================

@app.get("/complete-my-look/another/{item_id}")
def generate_another_complete_look(

    item_id: str,

    exclude_ids: str = ""

):

    wardrobe = load_wardrobe_data()

    preferences = load_user_preferences()


    # ========================================================
    # CHECK WARDROBE
    # ========================================================

    if not wardrobe:

        raise HTTPException(

            status_code=400,

            detail="Your wardrobe is empty."

        )


    # ========================================================
    # FIND SELECTED ITEM
    # ========================================================

    selected_item = None


    for item in wardrobe:

        if item.get("id") == item_id:

            selected_item = item

            break


    if selected_item is None:

        raise HTTPException(

            status_code=404,

            detail="Selected wardrobe item not found."

        )


    # ========================================================
    # READ EXCLUDED ITEM IDS
    # ========================================================

    excluded_id_list = [

        item_id.strip()

        for item_id in exclude_ids.split(",")

        if item_id.strip()

    ]


    excluded_items = {

        "bottom": [],
        "top": [],
        "footwear": [],
        "bag": [],
        "jewellery": [],
        "sunglasses": [],
        "watch": [],
        "accessory": []

    }


    # ========================================================
    # FIND ITEMS TO EXCLUDE
    # ========================================================

    for item in wardrobe:

        if item.get("id") not in excluded_id_list:

            continue


        if is_bottom(item):

            excluded_items[
                "bottom"
            ].append(
                item.get("id")
            )


        elif is_top(item):

            excluded_items[
                "top"
            ].append(
                item.get("id")
            )


        elif is_footwear(item):

            excluded_items[
                "footwear"
            ].append(
                item.get("id")
            )


        elif is_accessory(item):

            accessory_category = accessory_type(item)


            if accessory_category == "bag":

                excluded_items[
                    "bag"
                ].append(
                    item.get("id")
                )


            elif accessory_category == "jewellery":

                excluded_items[
                    "jewellery"
                ].append(
                    item.get("id")
                )


            elif accessory_category == "sunglasses":

                excluded_items[
                    "sunglasses"
                ].append(
                    item.get("id")
                )


            elif accessory_category == "watch":

                excluded_items[
                    "watch"
                ].append(
                    item.get("id")
                )


            else:

                excluded_items[
                    "accessory"
                ].append(
                    item.get("id")
                )


    # ========================================================
    # GENERATE ANOTHER LOOK
    # ========================================================

    complete_look = build_complete_look(

        selected_item,

        wardrobe,

        preferences,

        excluded_items

    )


    # ========================================================
    # CHECK RESULT
    # ========================================================

    if not complete_look["items"]:

        raise HTTPException(

            status_code=400,

            detail=
                "No different compatible items "
                "are available in your wardrobe."

        )


    return {

        "status":
            "success",

        "message":
            "Another complete look generated successfully.",

        "selected_item":
            complete_look["selected_item"],

        "items":
            complete_look["items"],

        "score":
            complete_look["score"],

        "preferences":
            preferences,

        "explanation":
            complete_look["explanation"]

    }


# ============================================================
# MAIN RECOMMENDATION
# ============================================================

@app.get("/recommendation")
def generate_recommendation():

    global RECOMMENDATION_INDEX

    global RECOMMENDATION_HISTORY


    wardrobe = load_wardrobe_data()

    preferences = load_user_preferences()


    # ========================================================
    # RESET RECOMMENDATION HISTORY
    # ========================================================

    RECOMMENDATION_INDEX = 0

    RECOMMENDATION_HISTORY = []


    # ========================================================
    # CHECK WARDROBE
    # ========================================================

    if not wardrobe:

        raise HTTPException(

            status_code=400,

            detail="Your wardrobe is empty."

        )


    # ========================================================
    # CHECK PREFERENCES
    # ========================================================

    if not preferences:

        raise HTTPException(

            status_code=400,

            detail="Please save your preferences first."

        )


    # ========================================================
    # CATEGORIES
    # ========================================================

    tops = [

        item

        for item in wardrobe

        if is_top(item)

    ]


    bottoms = [

        item

        for item in wardrobe

        if is_bottom(item)

    ]


    footwear = [

        item

        for item in wardrobe

        if is_footwear(item)

    ]


    accessories = [

        item

        for item in wardrobe

        if is_accessory(item)

    ]


    one_piece_items = [

        item

        for item in wardrobe

        if is_one_piece(item)

    ]


    # ========================================================
    # FIND BEST OUTFIT
    # ========================================================

    selected = find_best_outfit(

        tops,

        bottoms,

        footwear,

        accessories,

        one_piece_items,

        preferences

    )


    if not selected:

        raise HTTPException(

            status_code=400,

            detail=
                "Not enough wardrobe items "
                "to create an outfit."

        )


    # ========================================================
    # BUILD OUTFIT
    # ========================================================

    outfit = {}


    for category in [

        "one_piece",
        "top",
        "bottom",
        "footwear",
        "accessory"

    ]:

        result = selected.get(
            category
        )


        if result:

            outfit[category] = (

                result["item"]

            )


    # ========================================================
    # RECORD RECOMMENDATION
    # ========================================================

    RECOMMENDATION_HISTORY.append(

        outfit_signature(outfit)

    )


    # ========================================================
    # BUILD EXPLANATION
    # ========================================================

    explanation = build_outfit_explanation(

        selected,

        preferences

    )


    # ========================================================
    # BUDGET NOTE
    # ========================================================

    if preferences.get("budget"):

        explanation.append(

            "Budget preference was recorded, "
            "but wardrobe items currently do not "
            "contain price data, so budget was not "
            "used as a hard price filter."

        )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "status":
            "success",

        "message":
            "Compatibility-aware personalized "
            "outfit generated successfully.",

        "preferences":
            preferences,

        "outfit":
            outfit,

        "score":
            selected["score"],

        "explanation":
            explanation

    }


# ============================================================
# GENERATE ANOTHER OUTFIT
# ============================================================

@app.get("/recommendation/another")
def generate_another_recommendation():

    global RECOMMENDATION_INDEX

    global RECOMMENDATION_HISTORY


    wardrobe = load_wardrobe_data()

    preferences = load_user_preferences()


    # ========================================================
    # CHECK WARDROBE
    # ========================================================

    if not wardrobe:

        raise HTTPException(

            status_code=400,

            detail="Your wardrobe is empty."

        )


    # ========================================================
    # CHECK PREFERENCES
    # ========================================================

    if not preferences:

        raise HTTPException(

            status_code=400,

            detail="Please save your preferences first."

        )


    # ========================================================
    # CATEGORIES
    # ========================================================

    tops = [

        item

        for item in wardrobe

        if is_top(item)

    ]


    bottoms = [

        item

        for item in wardrobe

        if is_bottom(item)

    ]


    footwear = [

        item

        for item in wardrobe

        if is_footwear(item)

    ]


    accessories = [

        item

        for item in wardrobe

        if is_accessory(item)

    ]


    one_piece_items = [

        item

        for item in wardrobe

        if is_one_piece(item)

    ]


    # Keep Generate Another consistent with the main
    # recommendation by preferring the selected occasion.
    preferred_occasion = preferences.get("occasion")

    occasion_tops = [
        item
        for item in tops
        if occasion_relationship(
            item.get("occasion"),
            preferred_occasion
        ) >= 18
    ]

    occasion_bottoms = [
        item
        for item in bottoms
        if occasion_relationship(
            item.get("occasion"),
            preferred_occasion
        ) >= 18
    ]

    occasion_one_piece = [
        item
        for item in one_piece_items
        if occasion_relationship(
            item.get("occasion"),
            preferred_occasion
        ) >= 18
    ]

    if occasion_one_piece and not (
        occasion_tops and occasion_bottoms
    ):
        tops = []
        bottoms = []
        one_piece_items = occasion_one_piece
    else:
        tops = occasion_tops if occasion_tops else tops
        bottoms = occasion_bottoms if occasion_bottoms else bottoms
        one_piece_items = (
            occasion_one_piece
            if occasion_one_piece
            else one_piece_items
        )

    footwear = filter_items_by_occasion(
        footwear,
        preferred_occasion,
        fallback_to_all=False
    )
    accessories = filter_items_by_occasion(
        accessories,
        preferred_occasion,
        fallback_to_all=False
    )


    # ========================================================
    # GENERATE ALL CANDIDATES
    # ========================================================

    all_combinations = []


    # ========================================================
    # ONE-PIECE OUTFITS
    # ========================================================

    for one_piece in one_piece_items:

        one_piece_result = score_item(

            one_piece,

            preferences

        )


        shoes_to_use = (

            footwear
            if footwear
            else
            [None]

        )


        accessories_to_use = (

            accessories
            if accessories
            else
            [None]

        )


        for shoe in shoes_to_use:

            shoe_result = (

                score_item(

                    shoe,

                    preferences

                )

                if shoe

                else

                None

            )


            for accessory in accessories_to_use:

                accessory_result = (

                    score_item(

                        accessory,

                        preferences

                    )

                    if accessory

                    else

                    None

                )


                results = [

                    one_piece_result,

                    shoe_result,

                    accessory_result

                ]


                score = calculate_complete_outfit_score(

                    results,

                    preferences

                )


                all_combinations.append({

                    "one_piece":
                        one_piece_result,

                    "top":
                        None,

                    "bottom":
                        None,

                    "footwear":
                        shoe_result,

                    "accessory":
                        accessory_result,

                    "score":
                        score

                })


    # ========================================================
    # TOP + BOTTOM OUTFITS
    # ========================================================

    for top in tops:

        top_result = score_item(

            top,

            preferences

        )


        for bottom in bottoms:

            bottom_result = score_item(

                bottom,

                preferences

            )


            shoes_to_use = (

                footwear
                if footwear
                else
                [None]

            )


            accessories_to_use = (

                accessories
                if accessories
                else
                [None]

            )


            for shoe in shoes_to_use:

                shoe_result = (

                    score_item(

                        shoe,

                        preferences

                    )

                    if shoe

                    else

                    None

                )


                for accessory in accessories_to_use:

                    accessory_result = (

                        score_item(

                            accessory,

                            preferences

                        )

                        if accessory

                        else

                        None

                    )


                    results = [

                        top_result,

                        bottom_result,

                        shoe_result,

                        accessory_result

                    ]


                    score = calculate_complete_outfit_score(

                        results,

                        preferences

                    )


                    all_combinations.append({

                        "one_piece":
                            None,

                        "top":
                            top_result,

                        "bottom":
                            bottom_result,

                        "footwear":
                            shoe_result,

                        "accessory":
                            accessory_result,

                        "score":
                            score

                    })


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    unique_combinations = []

    seen = set()


    for combination in all_combinations:

        outfit = {}


        for key in [

            "one_piece",
            "top",
            "bottom",
            "footwear",
            "accessory"

        ]:

            result = combination.get(key)


            if result:

                outfit[key] = result["item"]


        signature = outfit_signature(

            outfit

        )


        if signature not in seen:

            seen.add(signature)

            unique_combinations.append(

                combination

            )


    all_combinations = unique_combinations


    # ========================================================
    # REMOVE PREVIOUSLY SHOWN OUTFITS
    # ========================================================

    unseen_combinations = []


    for combination in all_combinations:

        outfit = {}


        for key in [

            "one_piece",
            "top",
            "bottom",
            "footwear",
            "accessory"

        ]:

            result = combination.get(key)


            if result:

                outfit[key] = result["item"]


        signature = outfit_signature(

            outfit

        )


        if signature not in RECOMMENDATION_HISTORY:

            unseen_combinations.append(

                combination

            )


    # ========================================================
    # NO NEW COMBINATIONS
    # ========================================================

    if not unseen_combinations:

        raise HTTPException(

            status_code=400,

            detail=
                "No different outfit combination is available "
                "for the current wardrobe and preferences. "
                "Add more compatible wardrobe items to generate "
                "another recommendation."

        )


    # ========================================================
    # SORT BY SCORE
    # ========================================================

    unseen_combinations.sort(

        key=lambda x:
            x["score"],

        reverse=True

    )


    # ========================================================
    # SELECT NEXT BEST UNSEEN OUTFIT
    # ========================================================

    selected = unseen_combinations[0]


    # ========================================================
    # BUILD OUTFIT
    # ========================================================

    outfit = {}


    for category in [

        "one_piece",
        "top",
        "bottom",
        "footwear",
        "accessory"

    ]:

        result = selected.get(
            category
        )


        if result:

            outfit[category] = (

                result["item"]

            )


    # ========================================================
    # RECORD HISTORY
    # ========================================================

    RECOMMENDATION_HISTORY.append(

        outfit_signature(outfit)

    )


    RECOMMENDATION_INDEX += 1


    # ========================================================
    # EXPLANATION
    # ========================================================

    explanation = build_outfit_explanation(

        selected,

        preferences

    )


    explanation.insert(

        1,

        "This is a different outfit combination "
        "generated from your available wardrobe."

    )


    # ========================================================
    # BUDGET NOTE
    # ========================================================

    if preferences.get("budget"):

        explanation.append(

            "Budget preference was recorded, "
            "but wardrobe items currently do not "
            "contain price data, so budget was not "
            "used as a hard price filter."

        )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "status":
            "success",

        "message":
            "Another personalized outfit "
            "generated successfully.",

        "preferences":
            preferences,

        "outfit":
            outfit,

        "score":
            selected["score"],

        "explanation":
            explanation

    }