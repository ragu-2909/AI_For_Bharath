# Design Document: Crop Advisory Agent

## Overview

The Crop Advisory Agent is a multi-tier agricultural decision support system that combines AI/ML models, real-time data integration, and accessible user interfaces to deliver hyper-local farming recommendations. The system architecture prioritizes offline capability, low-resource operation, and accessibility for rural users with varying literacy levels.

### Design Principles

1. **Accessibility First**: Voice interface, Telegram bot, and simple UI for users with limited digital literacy
2. **Offline Resilience**: Core features available without connectivity through intelligent caching
3. **Proactive Intelligence**: Predictive risk assessment rather than reactive problem-solving
4. **Hyper-Local Context**: Recommendations tailored to specific soil, weather, and regional conditions
5. **Incremental Value**: Each component provides standalone value while integrating into the larger system

### MVP Scope (48-hour Hackathon)

The MVP focuses on:
- Crop recommendation engine with basic soil/season inputs
- Weather-aware irrigation alerts
- Image-based pest detection (limited model)
- Telegram bot interface
- Simple crop calendar
- Basic market price display
- Notification system

Post-MVP enhancements:
- Voice interface with multi-language support
- Advanced offline capabilities
- Comprehensive pest detection model
- Detailed market analytics
- Regional language expansion

## Architecture

### System Architecture

The system follows a microservices-inspired architecture with clear separation between data ingestion, AI processing, and delivery channels:

```
┌─────────────────────────────────────────────────────────────┐
│                     Delivery Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Mobile App   │  │ Telegram Bot │  │ Voice API    │     │
│  │ (Flutter)    │  │              │  │ (Future)     │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   API Gateway Layer                         │
│              (FastAPI / Node.js)                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Authentication │ Rate Limiting │ Request Routing    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Advisory Engine Core                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │    Crop      │  │  Irrigation  │  │    Risk      │     │
│  │ Recommender  │  │   Planner    │  │  Assessor    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Market     │  │   Calendar   │  │    Pest      │     │
│  │  Insights    │  │   Manager    │  │  Detector    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Firebase/   │  │   Weather    │  │   Market     │     │
│  │   MongoDB    │  │     API      │  │   Price API  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │  Crop Data   │  │  ML Models   │                        │
│  │  Repository  │  │   Storage    │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

1. **User Request Flow**:
   - User submits query via Mobile App or Telegram Bot
   - API Gateway authenticates and routes request
   - Advisory Engine Core processes request using relevant components
   - Results cached locally and returned to user

2. **Proactive Alert Flow**:
   - Weather API continuously monitored for changes
   - Risk Assessor evaluates conditions against user profiles
   - Notification System triggers alerts via preferred channels
   - Calendar Manager updates schedules if needed

3. **Offline Operation Flow**:
   - Mobile App caches recent recommendations and calendars
   - User queries processed locally when possible
   - Pending actions queued for sync
   - Background sync when connectivity restored

## Components and Interfaces

### 1. Crop Recommender

**Purpose**: Analyze soil conditions, seasonal context, and location to recommend suitable crops.

**Inputs**:
- `SoilProfile`: { soilType, pH, nitrogen, phosphorus, potassium, organicMatter, moisture }
- `SeasonalContext`: { season, month, region, rainfall }
- `LocationData`: { latitude, longitude, district, state }

**Outputs**:
- `CropRecommendation[]`: Array of recommended crops with scores

**Algorithm**:
```
function recommendCrops(soil, season, location):
    // Load crop database filtered by region
    candidateCrops = getCropsByRegion(location.state)
    
    // Score each crop based on multiple factors
    for crop in candidateCrops:
        soilScore = calculateSoilCompatibility(crop, soil)
        seasonScore = calculateSeasonalFit(crop, season)
        marketScore = getMarketDemandScore(crop, location)
        
        crop.totalScore = (soilScore * 0.5) + (seasonScore * 0.3) + (marketScore * 0.2)
    
    // Rank and return top 3-5 crops
    rankedCrops = sortByScore(candidateCrops)
    return rankedCrops.slice(0, 5)

function calculateSoilCompatibility(crop, soil):
    // Check pH range
    phScore = isInRange(soil.pH, crop.optimalPH) ? 1.0 : 0.5
    
    // Check NPK levels
    nScore = soil.nitrogen >= crop.minNitrogen ? 1.0 : 0.6
    pScore = soil.phosphorus >= crop.minPhosphorus ? 1.0 : 0.6
    kScore = soil.potassium >= crop.minPotassium ? 1.0 : 0.6
    
    // Check soil type compatibility
    typeScore = crop.suitableSoilTypes.includes(soil.soilType) ? 1.0 : 0.3
    
    return average([phScore, nScore, pScore, kScore, typeScore])
```

**ML Model** (Optional Enhancement):
- Use scikit-learn Random Forest or XGBoost trained on historical crop yield data
- Features: soil parameters, weather patterns, location, previous crop
- Target: crop yield success rate

### 2. Irrigation Planner

**Purpose**: Generate irrigation schedules and alerts based on weather forecasts and crop water requirements.

**Inputs**:
- `CropData`: { cropType, growthStage, sowingDate }
- `WeatherForecast`: { temperature, rainfall, humidity, windSpeed }[]
- `SoilMoisture`: { currentLevel, fieldCapacity }

**Outputs**:
- `IrrigationSchedule`: { nextIrrigationDate, waterQuantity, reason }
- `IrrigationAlert`: { urgency, message, recommendedAction }

**Algorithm**:
```
function planIrrigation(crop, weather, soilMoisture):
    // Calculate crop water requirement based on growth stage
    cropWaterNeed = getCropWaterRequirement(crop.cropType, crop.growthStage)
    
    // Check upcoming rainfall
    forecastedRainfall = sumRainfall(weather, days=2)
    
    if forecastedRainfall > cropWaterNeed:
        return {
            action: "SKIP",
            reason: "Sufficient rainfall expected",
            nextCheck: 2 days
        }
    
    // Calculate soil moisture deficit
    moistureDeficit = crop.optimalMoisture - soilMoisture.currentLevel
    
    if moistureDeficit > crop.criticalThreshold:
        return {
            action: "IRRIGATE_URGENT",
            waterQuantity: calculateWaterVolume(moistureDeficit, fieldSize),
            reason: "Soil moisture below critical level"
        }
    
    // Calculate evapotranspiration
    ET = calculateEvapotranspiration(weather.temperature, weather.humidity, crop)
    
    daysUntilIrrigation = (soilMoisture.currentLevel - crop.criticalThreshold) / ET
    
    return {
        action: "SCHEDULE",
        nextIrrigationDate: today + daysUntilIrrigation,
        waterQuantity: calculateWaterVolume(cropWaterNeed, fieldSize),
        reason: "Scheduled based on ET and soil moisture"
    }
```

### 3. Pest Detector

**Purpose**: Identify pests and diseases from crop/leaf images using computer vision.

**Inputs**:
- `Image`: Binary image data (JPEG/PNG)
- `CropType`: Optional context for better accuracy

**Outputs**:
- `PestDetection`: { pestName, confidence, severity, treatments[] }

**ML Model Architecture**:
```
Model: Convolutional Neural Network (CNN)
Base: MobileNetV2 or EfficientNet-Lite (optimized for mobile)
Training Data: PlantVillage dataset + India-specific pest images

Architecture:
- Input: 224x224x3 RGB image
- Base Model: MobileNetV2 (pre-trained on ImageNet)
- Custom Layers:
  - GlobalAveragePooling2D
  - Dense(256, activation='relu')
  - Dropout(0.3)
  - Dense(num_classes, activation='softmax')

Output Classes (MVP):
- Healthy
- Bacterial Blight
- Leaf Spot
- Aphids
- Caterpillar
- Fungal Infection
- Nutrient Deficiency
- Unknown/Unclear
```

**Processing Pipeline**:
```
function detectPest(image, cropType):
    // Validate image quality
    if not isValidImage(image):
        return { error: "INVALID_IMAGE", guidance: "Please capture a clear, well-lit image" }
    
    // Preprocess image
    processedImage = resizeAndNormalize(image, targetSize=224)
    
    // Run inference
    predictions = pestDetectionModel.predict(processedImage)
    topPrediction = getTopPrediction(predictions)
    
    if topPrediction.confidence < 0.6:
        return {
            status: "UNCERTAIN",
            message: "Unable to identify with confidence. Please consult local expert.",
            possibleIssues: getTopN(predictions, 3)
        }
    
    // Get treatment recommendations
    treatments = getTreatmentDatabase(topPrediction.pestName)
    
    return {
        pestName: topPrediction.pestName,
        confidence: topPrediction.confidence,
        severity: assessSeverity(image, topPrediction),
        organicTreatments: treatments.organic,
        chemicalTreatments: treatments.chemical,
        preventiveMeasures: treatments.preventive
    }
```

### 4. Calendar Manager

**Purpose**: Generate and maintain crop-specific farming activity timelines.

**Inputs**:
- `CropType`: Selected crop
- `SowingDate`: Date when crop was/will be sown
- `Region`: Geographic location for regional practices

**Outputs**:
- `CropCalendar`: { activities[], milestones[], harvestDate }

**Algorithm**:
```
function generateCropCalendar(cropType, sowingDate, region):
    // Load crop-specific timeline template
    template = getCropTemplate(cropType, region)
    
    calendar = {
        cropType: cropType,
        sowingDate: sowingDate,
        activities: []
    }
    
    // Generate activities based on template
    for activity in template.activities:
        activityDate = sowingDate + activity.daysAfterSowing
        
        calendar.activities.push({
            name: activity.name,
            scheduledDate: activityDate,
            dateRange: [activityDate - 2, activityDate + 2],
            description: activity.description,
            priority: activity.priority,
            completed: false
        })
    
    // Calculate harvest date
    calendar.harvestDate = sowingDate + template.totalDuration
    
    // Add weather-dependent activities
    calendar = addWeatherDependentActivities(calendar)
    
    return calendar

function updateCalendarForWeather(calendar, weatherForecast):
    // Adjust activities based on weather
    for activity in calendar.activities:
        if activity.weatherDependent and not activity.completed:
            if isUnfavorableWeather(weatherForecast, activity.scheduledDate):
                activity.scheduledDate = findNextFavorableDate(weatherForecast, activity)
                activity.rescheduled = true
    
    return calendar
```

### 5. Risk Assessor

**Purpose**: Predict and identify agricultural risks before they cause damage.

**Inputs**:
- `WeatherForecast`: Extended forecast data
- `CropData`: Current crop information
- `HistoricalData`: Past pest outbreaks, weather patterns
- `RegionalAlerts`: Government/agricultural department alerts

**Outputs**:
- `RiskAssessment`: { risks[], overallSeverity, recommendations[] }

**Algorithm**:
```
function assessRisks(weather, crop, historical, regional):
    risks = []
    
    // Weather-based risks
    if detectDroughtRisk(weather):
        risks.push({
            type: "DROUGHT",
            severity: calculateDroughtSeverity(weather, crop),
            confidence: 0.85,
            preventiveActions: ["Increase irrigation frequency", "Apply mulch", "Consider drought-resistant varieties"]
        })
    
    if detectFloodRisk(weather):
        risks.push({
            type: "FLOOD",
            severity: calculateFloodSeverity(weather, crop.location),
            confidence: 0.90,
            preventiveActions: ["Ensure drainage", "Harvest early if possible", "Protect stored produce"]
        })
    
    if detectFrostRisk(weather, crop):
        risks.push({
            type: "FROST",
            severity: "HIGH",
            confidence: 0.95,
            preventiveActions: ["Cover crops", "Light fires in field", "Spray water before sunrise"]
        })
    
    // Pest outbreak prediction
    pestRisk = predictPestOutbreak(weather, crop, historical)
    if pestRisk.probability > 0.6:
        risks.push({
            type: "PEST_OUTBREAK",
            pestType: pestRisk.likelyPest,
            severity: pestRisk.severity,
            confidence: pestRisk.probability,
            preventiveActions: pestRisk.preventiveMeasures
        })
    
    // Regional alerts integration
    for alert in regional:
        if alert.appliesToCrop(crop.cropType):
            risks.push(convertRegionalAlert(alert))
    
    // Prioritize risks
    risks = sortBySeverityAndImmediacy(risks)
    
    return {
        risks: risks,
        overallSeverity: calculateOverallSeverity(risks),
        immediateActions: getImmediateActions(risks)
    }
```

### 6. Market Insights

**Purpose**: Provide current market prices and trends for informed crop selection.

**Inputs**:
- `CropType`: Crop to query
- `Location`: Market location (mandi)
- `DateRange`: Period for trend analysis

**Outputs**:
- `MarketData`: { currentPrice, priceHistory[], trend, forecast }

**Algorithm**:
```
function getMarketInsights(cropType, location, dateRange):
    // Fetch current prices from API
    currentPrice = fetchMarketPrice(cropType, location)
    
    // Get historical prices
    priceHistory = fetchPriceHistory(cropType, location, dateRange)
    
    // Calculate trend
    trend = calculateTrend(priceHistory)
    
    // Simple forecast (moving average)
    forecast = calculateMovingAverage(priceHistory, window=7)
    
    // Identify favorable pricing
    isFavorable = currentPrice > average(priceHistory) * 1.1
    
    return {
        cropType: cropType,
        currentPrice: currentPrice,
        unit: "per quintal",
        priceHistory: priceHistory,
        trend: trend, // "RISING", "FALLING", "STABLE"
        forecast: forecast,
        isFavorable: isFavorable,
        lastUpdated: timestamp
    }
```

### 7. Notification System

**Purpose**: Deliver timely alerts through multiple channels while preventing alert fatigue.

**Inputs**:
- `Alert`: { type, severity, message, targetUsers[] }
- `UserPreferences`: { channels[], quietHours, maxDailyAlerts }

**Outputs**:
- `DeliveryStatus`: { sent, failed, queued }

**Algorithm**:
```
function sendNotification(alert, user):
    // Check alert fatigue limits
    todayAlertCount = getAlertCount(user, today)
    if todayAlertCount >= user.maxDailyAlerts and alert.severity != "CRITICAL":
        return queueForTomorrow(alert, user)
    
    // Check quiet hours
    if isQuietHours(user.quietHours) and alert.severity != "CRITICAL":
        return queueForLater(alert, user)
    
    // Determine delivery channels
    channels = selectChannels(user.preferences, alert.severity)
    
    results = []
    for channel in channels:
        if channel == "PUSH":
            results.push(sendPushNotification(alert, user))
        else if channel == "SMS":
            results.push(sendSMS(alert, user))
        else if channel == "TELEGRAM":
            results.push(sendTelegramMessage(alert, user))
    
    // Log delivery
    logNotification(alert, user, results)
    
    return aggregateResults(results)

function selectChannels(preferences, severity):
    if severity == "CRITICAL":
        return ["PUSH", "SMS", "TELEGRAM"] // All channels
    else if severity == "HIGH":
        return preferences.primaryChannels
    else:
        return [preferences.defaultChannel]
```

### 8. Voice Interface (Post-MVP)

**Purpose**: Enable voice-based interaction for users with limited literacy.

**Inputs**:
- `AudioData`: Voice recording
- `Language`: User's preferred language

**Outputs**:
- `TranscribedText`: Recognized text
- `SpokenResponse`: Audio response

**Architecture**:
```
Speech-to-Text: Google Cloud Speech-to-Text API or Azure Speech Services
- Support for Hindi, English, Tamil, Telugu, Marathi
- Noise reduction and accent adaptation

Text-to-Speech: Google Cloud Text-to-Speech or Azure TTS
- Natural-sounding voices in regional languages
- Adjustable speech rate for clarity

Processing Flow:
1. Receive audio from user
2. Detect language (if not specified)
3. Transcribe to text using STT
4. Process query through Advisory Engine
5. Generate text response
6. Convert to speech using TTS
7. Return audio response
```

### 9. Telegram Bot Interface

**Purpose**: Provide full system access through Telegram for users without smartphones.

**Commands**:
- `/start` - Register and set preferences
- `/recommend` - Get crop recommendations
- `/weather` - Get weather and irrigation advice
- `/calendar` - View crop calendar
- `/market` - Check market prices
- `/help` - Get help and usage instructions

**Image Handling**:
- User sends image → Bot processes through Pest Detector → Returns identification and treatment

**Implementation**:
```
Bot Framework: python-telegram-bot or Telegraf (Node.js)

function handleMessage(message, user):
    if message.isCommand():
        return handleCommand(message.command, user)
    else if message.hasImage():
        return handleImageAnalysis(message.image, user)
    else if message.isText():
        return handleTextQuery(message.text, user)

function handleCommand(command, user):
    switch command:
        case "/recommend":
            return getCropRecommendations(user.profile)
        case "/weather":
            return getWeatherAndIrrigation(user.location)
        case "/calendar":
            return getCropCalendar(user.currentCrop)
        case "/market":
            return getMarketPrices(user.location)
        default:
            return "Unknown command. Type /help for assistance."
```

## Data Models

### User Profile
```typescript
interface UserProfile {
    userId: string;
    name: string;
    phoneNumber: string;
    language: "hi" | "en" | "ta" | "te" | "mr";
    location: {
        latitude: number;
        longitude: number;
        district: string;
        state: string;
    };
    farmDetails: {
        landSize: number; // in acres
        soilType: string;
        irrigationType: "rainfed" | "drip" | "sprinkler" | "flood";
    };
    preferences: {
        notificationChannels: ("push" | "sms" | "telegram")[];
        quietHours: { start: string; end: string };
        maxDailyAlerts: number;
    };
    currentCrops: CropInstance[];
    createdAt: Date;
    lastActive: Date;
}
```

### Soil Profile
```typescript
interface SoilProfile {
    soilType: "clay" | "sandy" | "loamy" | "silt" | "red" | "black";
    pH: number; // 0-14
    nitrogen: number; // kg/ha
    phosphorus: number; // kg/ha
    potassium: number; // kg/ha
    organicMatter: number; // percentage
    moisture: number; // percentage
    testDate: Date;
    location: string;
}
```

### Crop Recommendation
```typescript
interface CropRecommendation {
    cropName: string;
    scientificName: string;
    suitabilityScore: number; // 0-1
    expectedYield: { min: number; max: number; unit: string };
    growingDuration: number; // days
    waterRequirement: "low" | "medium" | "high";
    marketDemand: "low" | "medium" | "high";
    estimatedRevenue: { min: number; max: number; currency: string };
    reasons: string[]; // Why this crop is recommended
}
```

### Crop Instance
```typescript
interface CropInstance {
    instanceId: string;
    cropType: string;
    sowingDate: Date;
    expectedHarvestDate: Date;
    fieldLocation: { latitude: number; longitude: number };
    fieldSize: number; // acres
    growthStage: "sowing" | "germination" | "vegetative" | "flowering" | "fruiting" | "maturity";
    calendar: CropCalendar;
    healthStatus: "healthy" | "at-risk" | "diseased";
    lastUpdated: Date;
}
```

### Crop Calendar
```typescript
interface CropCalendar {
    cropType: string;
    sowingDate: Date;
    harvestDate: Date;
    activities: CalendarActivity[];
}

interface CalendarActivity {
    activityId: string;
    name: string;
    description: string;
    scheduledDate: Date;
    dateRange: { start: Date; end: Date };
    priority: "low" | "medium" | "high" | "critical";
    category: "irrigation" | "fertilization" | "pest-control" | "weeding" | "harvest";
    completed: boolean;
    completedDate?: Date;
    weatherDependent: boolean;
    rescheduled: boolean;
}
```

### Weather Context
```typescript
interface WeatherContext {
    location: { latitude: number; longitude: number };
    current: {
        temperature: number; // Celsius
        humidity: number; // percentage
        rainfall: number; // mm
        windSpeed: number; // km/h
        conditions: string;
    };
    forecast: WeatherForecast[];
    lastUpdated: Date;
}

interface WeatherForecast {
    date: Date;
    temperatureMin: number;
    temperatureMax: number;
    humidity: number;
    rainfallProbability: number; // percentage
    expectedRainfall: number; // mm
    conditions: string;
}
```

### Pest Detection Result
```typescript
interface PestDetectionResult {
    detectionId: string;
    imageUrl: string;
    cropType?: string;
    detectedIssue: {
        name: string;
        category: "pest" | "disease" | "nutrient-deficiency" | "healthy";
        confidence: number; // 0-1
        severity: "low" | "medium" | "high" | "critical";
    };
    treatments: {
        organic: Treatment[];
        chemical: Treatment[];
        preventive: string[];
    };
    detectedAt: Date;
}

interface Treatment {
    name: string;
    description: string;
    application: string;
    dosage: string;
    timing: string;
    precautions: string[];
}
```

### Risk Assessment
```typescript
interface RiskAssessment {
    assessmentId: string;
    userId: string;
    cropInstanceId: string;
    assessmentDate: Date;
    risks: Risk[];
    overallSeverity: "low" | "medium" | "high" | "critical";
    immediateActions: string[];
}

interface Risk {
    riskType: "drought" | "flood" | "frost" | "pest-outbreak" | "disease" | "market-crash";
    severity: "low" | "medium" | "high" | "critical";
    confidence: number; // 0-1
    description: string;
    preventiveActions: string[];
    timeframe: string; // "immediate", "24-48 hours", "next week"
    affectedCrops: string[];
}
```

### Market Data
```typescript
interface MarketData {
    cropType: string;
    market: string; // Mandi name
    currentPrice: number;
    unit: string; // "per quintal"
    priceHistory: PricePoint[];
    trend: "rising" | "falling" | "stable";
    forecast: number;
    isFavorable: boolean;
    lastUpdated: Date;
}

interface PricePoint {
    date: Date;
    price: number;
    volume?: number; // Optional: trading volume
}
```

### Notification
```typescript
interface Notification {
    notificationId: string;
    userId: string;
    type: "weather-alert" | "irrigation-reminder" | "calendar-activity" | "risk-warning" | "market-update";
    severity: "low" | "medium" | "high" | "critical";
    title: string;
    message: string;
    actionable: boolean;
    actionUrl?: string;
    channels: ("push" | "sms" | "telegram")[];
    sentAt: Date;
    readAt?: Date;
    expiresAt?: Date;
}
```



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Crop Recommender Properties

**Property 1: Minimum Recommendations Count**
*For any* valid Soil_Profile and seasonal context, the Crop_Recommender should return a ranked list containing at least 3 suitable crops.
**Validates: Requirements 1.1**

**Property 2: Recommendation Scoring Factors**
*For any* two crops where one has better soil compatibility (pH, NPK, soil type match) than the other, the better-matched crop should receive a higher suitability score when all other factors are equal.
**Validates: Requirements 1.2**

**Property 3: Market Demand Tie-Breaking**
*For any* two crops with suitability scores within 0.05 of each other, the crop with higher market demand should be ranked higher in the recommendation list.
**Validates: Requirements 1.3**

**Property 4: Recommendation Output Completeness**
*For any* crop recommendation in the output list, it should include expectedYield (with min and max), growingDuration, and all other required fields defined in the CropRecommendation data model.
**Validates: Requirements 1.5**

### Irrigation Planner Properties

**Property 5: Rainfall Delay Logic**
*For any* Weather_Context where forecasted rainfall within 48 hours exceeds the crop's daily water requirement, the Irrigation_Planner should recommend delaying irrigation.
**Validates: Requirements 2.1**

**Property 6: Moisture Threshold Alerting**
*For any* soil moisture level below the crop-specific critical threshold, the Irrigation_Planner should generate an irrigation alert with urgency marked as high or critical.
**Validates: Requirements 2.2**

**Property 7: Multi-Factor Irrigation Sensitivity**
*For any* irrigation schedule, changing the crop type, growth stage, soil type, or weather forecast should result in a different irrigation recommendation (demonstrating all factors are considered).
**Validates: Requirements 2.3**

**Property 8: Extreme Weather Protective Measures**
*For any* Weather_Context indicating extreme conditions (temperature >40°C or <5°C, wind speed >50 km/h, or heavy rainfall >100mm), the Irrigation_Planner should include protective measures in addition to irrigation guidance.
**Validates: Requirements 2.4**

**Property 9: Irrigation Output Completeness**
*For any* irrigation recommendation, it should specify waterQuantity (in liters or mm) and timing (date/time or relative timing like "within 24 hours").
**Validates: Requirements 2.5**

### Pest Detector Properties

**Property 10: Pest Detection Output Structure**
*For any* pest or disease detection result, it should include confidence score (0-1), severity assessment (low/medium/high/critical), and treatment recommendations with both organic and chemical options.
**Validates: Requirements 3.2, 3.5**

### Voice Interface Properties

**Property 11: Text-to-Speech Language Consistency**
*For any* Advisory_Engine response and user language preference, the Voice_Interface should generate speech output in the user's preferred language.
**Validates: Requirements 4.2**

### Calendar Manager Properties

**Property 12: Complete Calendar Generation**
*For any* valid crop type and sowing date, the Calendar_Manager should generate a Crop_Calendar that includes at least one activity from each category: sowing, irrigation, fertilization, pest control, and harvest.
**Validates: Requirements 5.1, 5.2**

**Property 13: Activity Due Date Notification Triggering**
*For any* Crop_Calendar where an activity's scheduled date matches the current date (within the date range), the Calendar_Manager should trigger a reminder notification for that activity.
**Validates: Requirements 5.3**

**Property 14: Weather-Based Calendar Updates**
*For any* Crop_Calendar with weather-dependent activities, when weather conditions become unfavorable for a scheduled activity, the Calendar_Manager should update the activity's scheduled date and mark it as rescheduled.
**Validates: Requirements 5.4**

**Property 15: Calendar Chronological Ordering**
*For any* generated Crop_Calendar, all activities should be ordered chronologically by scheduledDate, and each activity should have a valid dateRange with start <= end.
**Validates: Requirements 5.5**

### Market Insights Properties

**Property 16: Market Price Integration with Recommendations**
*For any* crop recommendation list, each recommended crop should have associated market data including currentPrice and unit.
**Validates: Requirements 6.1**

**Property 17: Historical Price Data Completeness**
*For any* market data display where data is available, the priceHistory should contain at least 30 days of price points (or indicate if fewer days are available due to data limitations).
**Validates: Requirements 6.2**

**Property 18: Favorable Trend Highlighting**
*For any* crop where the current price is more than 15% above the 30-day average price, the Market_Insights should mark it as having a favorable pricing trend.
**Validates: Requirements 6.3**

### Notification System Properties

**Property 19: Severe Weather Alert Triggering**
*For any* Weather_Context indicating severe weather (heavy rainfall >100mm, extreme temperature, or storm warnings) within 24 hours, the Notification_System should send an alert to affected farmers.
**Validates: Requirements 7.1**

**Property 20: Calendar Activity Reminder Triggering**
*For any* Crop_Calendar activity scheduled within 24 hours that is not yet completed, the Notification_System should send a reminder notification.
**Validates: Requirements 7.2**

**Property 21: Pest Risk Advisory Triggering**
*For any* Risk_Assessment where pest outbreak risk severity is high or critical, the Notification_System should send a preventive advisory notification.
**Validates: Requirements 7.3**

**Property 22: Notification Channel Preference Respect**
*For any* notification sent to a user, it should be delivered through the user's preferred notification channels as specified in their UserProfile preferences (unless overridden for critical alerts).
**Validates: Requirements 7.4, 9.4**

**Property 23: Daily Notification Rate Limiting**
*For any* user on a given day, the Notification_System should not send more than the user's maxDailyAlerts (default 5) non-critical notifications.
**Validates: Requirements 7.5**

### Dashboard Properties

**Property 24: Dashboard Recommendations Presence**
*For any* user opening the application, the dashboard should display at least one recommendation or status message for the current day.
**Validates: Requirements 8.1**

**Property 25: Dashboard Component Completeness**
*For any* dashboard display, it should include all required components: weather summary, irrigation status, pending tasks list, and active alerts list (even if some lists are empty).
**Validates: Requirements 8.2**

**Property 26: Urgent Action Highlighting**
*For any* dashboard where there are actions with priority marked as "high" or "critical", those actions should be visually distinguished (flagged/highlighted) from normal priority actions.
**Validates: Requirements 8.3**

### Telegram Bot Properties

**Property 27: Telegram Image Analysis**
*For any* image sent to the Telegram bot, the Pest_Detector should analyze it and return a PestDetectionResult with the same structure as the mobile app response.
**Validates: Requirements 9.2**

**Property 28: Telegram Response Formatting**
*For any* recommendation or advisory sent via Telegram, the message length should not exceed 4096 characters (Telegram's limit), and long content should be split into multiple messages.
**Validates: Requirements 9.3**

### Offline Capability Properties

**Property 29: Offline Cached Data Access**
*For any* user in offline mode, the Advisory_Engine should provide access to the most recently cached Crop_Calendar and crop recommendations without requiring network connectivity.
**Validates: Requirements 10.1**

**Property 30: Offline Action Queuing**
*For any* user action performed while offline (e.g., marking calendar activity as complete), the action should be queued locally and automatically synced when connectivity is restored.
**Validates: Requirements 10.2**

**Property 31: Offline Limitation Indication**
*For any* feature that requires network access (e.g., real-time weather, pest detection, market prices), when accessed in offline mode, the Advisory_Engine should display a clear message indicating the feature is unavailable offline.
**Validates: Requirements 10.3**

**Property 32: Automatic Sync on Reconnection**
*For any* user transitioning from offline to online mode, the Advisory_Engine should automatically sync all queued actions and refresh time-sensitive data (weather, market prices) within 30 seconds of reconnection.
**Validates: Requirements 10.4**

**Property 33: Minimum Cache Duration**
*For any* user's cached data, it should include at least 7 days of weather forecasts and the complete Crop_Calendar for all active crops.
**Validates: Requirements 10.5**

### Risk Assessor Properties

**Property 34: Multi-Risk Type Identification**
*For any* combination of Weather_Context and crop data that meets the criteria for drought (no rainfall forecast for 7+ days), flood (rainfall >150mm in 24 hours), frost (temperature <5°C), or pest outbreak (high humidity + warm temperature), the Risk_Assessor should identify the corresponding risk type.
**Validates: Requirements 11.1**

**Property 35: Risk Assessment Output Completeness**
*For any* identified risk, the Risk_Assessment should include both a severity score (low/medium/high/critical) and a confidence level (0-1).
**Validates: Requirements 11.2, 11.5**

**Property 36: Medium+ Risk Preventive Actions**
*For any* risk with severity of medium, high, or critical, the Risk_Assessment should include at least one specific preventive action recommendation.
**Validates: Requirements 11.3**

**Property 37: Risk Prioritization Ordering**
*For any* Risk_Assessment containing multiple risks, the risks array should be ordered with critical severity first, then high, then medium, then low; and within the same severity level, ordered by immediacy (timeframe).
**Validates: Requirements 11.4**

### Data Privacy and Security Properties

**Property 38: Analytics Data Anonymization**
*For any* Soil_Profile or farm location data stored for analytics purposes, it should not contain userId, name, phoneNumber, or any other personally identifiable information.
**Validates: Requirements 12.2**

**Property 39: Authentication Requirement for Personalized Features**
*For any* request to access personalized features (user profile, crop calendars, notifications), the Advisory_Engine should reject the request if no valid authentication token is provided.
**Validates: Requirements 12.5**

### Localization and Accessibility Properties

**Property 40: Language Preference Application**
*For any* user with a preferred language set in their UserProfile, all text content displayed by the Advisory_Engine should be in that language (or indicate if translation is unavailable for specific content).
**Validates: Requirements 13.1**

**Property 41: Local Unit Usage**
*For any* measurement displayed by the Advisory_Engine (land area, crop yield, water quantity), it should use Indian local units: acres for area, quintals for weight, liters for volume.
**Validates: Requirements 13.2**

**Property 42: Region-Specific Recommendations**
*For any* crop recommendation generated for a user in a specific state/district, the recommended crops should be from the crop database filtered for that region, ensuring regional appropriateness.
**Validates: Requirements 13.3**

### System Error Handling Properties

**Property 43: Error Logging and User-Friendly Messages**
*For any* system error or exception that occurs during request processing, the Advisory_Engine should log the error with timestamp and context, and return a user-friendly error message (not technical stack traces) to the user.
**Validates: Requirements 14.5**

### Data Validation Properties

**Property 44: Weather Data Staleness Detection**
*For any* Weather_Context where the lastUpdated timestamp is more than 6 hours old, the Advisory_Engine should indicate data staleness and prompt for a refresh before using it for critical decisions.
**Validates: Requirements 15.3**

**Property 45: Input Data Validation**
*For any* input data (Soil_Profile, Weather_Context, CropInstance), the Advisory_Engine should validate that all required fields are present and values are within valid ranges before processing, rejecting invalid inputs with specific error messages.
**Validates: Requirements 15.4**

**Property 46: Recommendation Source Attribution**
*For any* recommendation or advisory generated by the Advisory_Engine, it should include source attribution indicating which data sources (weather API, market API, crop database, ML model) were used to generate the insight.
**Validates: Requirements 15.5**

## Error Handling

### Error Categories

1. **Input Validation Errors**
   - Invalid soil parameters (pH out of range, negative nutrient values)
   - Missing required fields in user input
   - Invalid date ranges or crop types
   - Response: Return 400 Bad Request with specific field-level error messages

2. **External Service Failures**
   - Weather API unavailable or timeout
   - Market price API failure
   - ML model inference errors
   - Response: Use cached data when available, return 503 Service Unavailable with retry guidance

3. **Authentication and Authorization Errors**
   - Invalid or expired authentication tokens
   - Unauthorized access to other users' data
   - Response: Return 401 Unauthorized or 403 Forbidden with clear messages

4. **Resource Not Found Errors**
   - User profile not found
   - Crop calendar not found
   - Crop type not in database
   - Response: Return 404 Not Found with suggestions for valid resources

5. **Rate Limiting Errors**
   - Too many requests from a single user
   - API quota exceeded
   - Response: Return 429 Too Many Requests with retry-after header

6. **Data Consistency Errors**
   - Stale weather data
   - Conflicting calendar updates
   - Response: Indicate staleness, prompt for refresh, or resolve conflicts automatically

### Error Handling Strategies

**Graceful Degradation**:
- If weather API fails, use last cached forecast with staleness warning
- If market API fails, show last known prices with timestamp
- If ML model fails, fall back to rule-based recommendations

**Retry Logic**:
- Exponential backoff for transient failures (network timeouts)
- Maximum 3 retry attempts for external API calls
- Circuit breaker pattern for repeatedly failing services

**User Communication**:
- All error messages in user's preferred language
- Avoid technical jargon (no stack traces, error codes explained)
- Provide actionable guidance ("Try again in 5 minutes", "Check your internet connection")

**Logging and Monitoring**:
- Log all errors with severity levels (ERROR, WARNING, INFO)
- Include request context (userId, timestamp, endpoint, input parameters)
- Alert on critical errors (database failures, authentication system down)
- Track error rates and patterns for proactive fixes

### Offline Error Handling

When offline:
- Queue actions that require network (with user notification)
- Clearly indicate which features are unavailable
- Provide estimated time for sync when connectivity returns
- Prevent data loss by persisting queued actions locally

## Testing Strategy

### Dual Testing Approach

The Crop Advisory Agent requires both unit testing and property-based testing to ensure comprehensive correctness:

**Unit Tests**: Focus on specific examples, edge cases, and integration points
- Specific crop recommendation scenarios (e.g., rice in monsoon season with clay soil)
- Edge cases like extreme weather conditions, empty soil profiles, invalid inputs
- Integration between components (e.g., calendar manager triggering notifications)
- Error handling paths (API failures, invalid authentication)

**Property-Based Tests**: Verify universal properties across all inputs
- Generate random soil profiles, weather contexts, and crop data
- Verify properties hold for all generated inputs (100+ iterations per test)
- Catch edge cases that manual test cases might miss
- Ensure system behavior is consistent across the input space

### Property-Based Testing Configuration

**Framework Selection**:
- **Python**: Use `hypothesis` library for property-based testing
- **TypeScript/JavaScript**: Use `fast-check` library
- **Integration**: Configure CI/CD to run property tests on every commit

**Test Configuration**:
- Minimum 100 iterations per property test (due to randomization)
- Seed-based reproducibility for failed tests
- Shrinking enabled to find minimal failing examples
- Timeout: 30 seconds per property test

**Test Tagging**:
Each property-based test must reference its design document property:
```python
# Feature: crop-advisory-agent, Property 1: Minimum Recommendations Count
@given(soil_profile=soil_profiles(), season=seasons())
def test_minimum_recommendations_count(soil_profile, season):
    recommendations = crop_recommender.recommend(soil_profile, season)
    assert len(recommendations) >= 3
```

### Test Data Generators

**For Property-Based Tests**:
```python
# Hypothesis strategies for generating test data
@composite
def soil_profiles(draw):
    return SoilProfile(
        soilType=draw(sampled_from(["clay", "sandy", "loamy", "silt", "red", "black"])),
        pH=draw(floats(min_value=4.0, max_value=9.0)),
        nitrogen=draw(floats(min_value=0, max_value=500)),
        phosphorus=draw(floats(min_value=0, max_value=200)),
        potassium=draw(floats(min_value=0, max_value=300)),
        organicMatter=draw(floats(min_value=0, max_value=10)),
        moisture=draw(floats(min_value=0, max_value=100))
    )

@composite
def weather_contexts(draw):
    return WeatherContext(
        current={
            "temperature": draw(floats(min_value=-5, max_value=50)),
            "humidity": draw(floats(min_value=0, max_value=100)),
            "rainfall": draw(floats(min_value=0, max_value=300)),
            "windSpeed": draw(floats(min_value=0, max_value=100))
        },
        forecast=draw(lists(weather_forecasts(), min_size=1, max_size=7))
    )
```

### Unit Test Coverage Targets

- **Component-level**: 80% code coverage minimum
- **Integration**: All component interactions tested
- **Edge cases**: All error handling paths covered
- **API endpoints**: All endpoints with success and failure scenarios

### Testing Priorities for MVP (48-hour Hackathon)

**Must Test (Critical Path)**:
1. Crop recommendation with basic soil inputs
2. Irrigation alert generation based on weather
3. Pest detection image processing
4. Telegram bot message handling
5. Calendar generation for common crops
6. Notification delivery

**Should Test (Important but not blocking)**:
1. Market price integration
2. Risk assessment logic
3. Offline caching and sync
4. Dashboard data aggregation

**Nice to Test (Post-MVP)**:
1. Voice interface
2. Multi-language support
3. Advanced offline scenarios
4. Performance under load

### Continuous Integration

- Run unit tests on every commit (fast feedback)
- Run property tests on pull requests (comprehensive validation)
- Run integration tests nightly (catch system-level issues)
- Monitor test execution time and optimize slow tests
- Fail builds on test failures or coverage drops

