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

### AWS Technology Stack

The system leverages AWS services for scalability, reliability, and cost-effectiveness:

**Compute & API**:
- AWS Lambda - Serverless compute for API endpoints and background tasks
- Amazon API Gateway - RESTful API management with authentication
- AWS Fargate - Container orchestration for ML model serving

**AI/ML Services**:
- Amazon SageMaker - ML model training, deployment, and inference
- Amazon Rekognition - Image analysis for pest detection
- Amazon Bedrock - Foundation models for conversational AI
- Amazon Polly - Text-to-speech for voice interface
- Amazon Transcribe - Speech-to-text for voice queries

**Data Storage**:
- Amazon DynamoDB - NoSQL database for user profiles, activities, events
- Amazon S3 - Object storage for images, ML models, backups
- Amazon RDS (PostgreSQL) - Relational database for structured data
- Amazon ElastiCache (Redis) - Caching layer for weather, market data

**Messaging & Notifications**:
- Amazon SNS - Push notifications and SMS delivery
- Amazon SQS - Message queuing for async processing
- Amazon EventBridge - Event-driven architecture coordination

**Integration & Data**:
- AWS AppSync - GraphQL API for real-time data sync
- Amazon Kinesis - Real-time data streaming for sensor data
- AWS Glue - ETL for data processing and feature engineering

**Security & Auth**:
- Amazon Cognito - User authentication and authorization
- AWS Secrets Manager - API keys and credentials management
- AWS IAM - Access control and permissions

**Monitoring & Analytics**:
- Amazon CloudWatch - Logging, metrics, and alarms
- AWS X-Ray - Distributed tracing and performance analysis
- Amazon QuickSight - Business intelligence and dashboards

**Content Delivery**:
- Amazon CloudFront - CDN for static assets and API acceleration
- AWS Amplify - Mobile app hosting and CI/CD

**Voice & Bot**:
- Amazon Lex - Conversational bot framework
- Amazon Connect - Voice call handling (future)

## Architecture

### AI Agentic Workflow

The Crop Advisory Agent employs an AI agentic workflow that orchestrates multiple specialized agents to deliver intelligent, context-aware recommendations. This architecture enables autonomous decision-making, adaptive learning, and proactive advisory services.

**Agentic Architecture Overview**:

```
┌─────────────────────────────────────────────────────────────┐
│                  Orchestration Agent                        │
│         (Coordinates multi-agent workflows)                 │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Perception  │    │   Reasoning  │    │    Action    │
│    Agent     │    │    Agent     │    │    Agent     │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────────────────────────────────────────────┐
│              Specialized Sub-Agents                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │   Crop     │  │  Weather   │  │    Pest    │    │
│  │  Advisor   │  │  Analyst   │  │  Detector  │    │
│  └────────────┘  └────────────┘  └────────────┘    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │ Irrigation │  │   Market   │  │    Risk    │    │
│  │  Planner   │  │  Analyst   │  │  Assessor  │    │
│  └────────────┘  └────────────┘  └────────────┘    │
└──────────────────────────────────────────────────────┘
```

**Agent Roles and Responsibilities**:

1. **Orchestration Agent**:
   - Receives farmer queries and contextual data
   - Determines which specialized agents to invoke
   - Coordinates multi-step workflows (e.g., crop recommendation → irrigation planning → risk assessment)
   - Aggregates results from multiple agents into coherent recommendations
   - Maintains conversation context for follow-up queries

2. **Perception Agent**:
   - Processes multimodal inputs (text, voice, images)
   - Extracts structured data from farmer inputs
   - Integrates real-time data from sensors, weather APIs, and market feeds
   - Maintains situational awareness of farm conditions

3. **Reasoning Agent**:
   - Applies domain knowledge and ML models to analyze situations
   - Performs causal reasoning (e.g., "low yield due to nutrient deficiency")
   - Generates hypotheses and validates them against data
   - Learns from historical outcomes to improve recommendations

4. **Action Agent**:
   - Translates recommendations into actionable steps
   - Schedules notifications and reminders
   - Triggers automated responses (e.g., irrigation alerts)
   - Monitors action completion and outcomes

**Agentic Workflow Example - Crop Recommendation**:

```
Farmer Query: "What should I plant this season?"
        ↓
Orchestration Agent:
  - Identifies intent: crop recommendation
  - Gathers context: farmer profile, soil data, location, season
  - Invokes: Perception Agent → Reasoning Agent → Action Agent
        ↓
Perception Agent:
  - Retrieves farmer's soil profile from database
  - Fetches current weather and 30-day forecast
  - Queries market prices for potential crops
        ↓
Reasoning Agent:
  - Invokes Crop Advisor sub-agent with soil + weather + market data
  - Invokes Risk Assessor to identify potential threats
  - Ranks crops by suitability score
  - Generates explanation for each recommendation
        ↓
Action Agent:
  - Formats recommendations in farmer's language
  - Creates crop calendar for top recommendation
  - Schedules follow-up check-in notification
  - Logs recommendation for learning
        ↓
Response: "Based on your black soil and monsoon season, I recommend:
1. Cotton (85% match) - High market demand, good rainfall expected
2. Soybean (78% match) - Lower water needs, stable prices
3. Pigeon pea (72% match) - Drought-resistant, intercropping option"
```

**Agentic Learning Loop**:

The system implements a continuous learning cycle:
1. **Recommendation**: Agent provides advice based on current knowledge
2. **Action**: Farmer follows (or ignores) recommendation
3. **Outcome**: System tracks crop yield, pest incidents, irrigation efficiency
4. **Feedback**: Farmer provides explicit feedback or system infers from outcomes
5. **Learning**: Models updated with new data, improving future recommendations

**Multi-Agent Collaboration Patterns**:

- **Sequential**: Crop recommendation → Irrigation planning → Calendar generation
- **Parallel**: Weather analysis + Market analysis + Soil analysis (concurrent)
- **Hierarchical**: Orchestrator delegates to specialized agents, which may invoke sub-agents
- **Collaborative**: Risk Assessor consults Weather Analyst + Pest Detector + Market Analyst

### System Architecture

The system follows a serverless microservices architecture on AWS with clear separation between data ingestion, AI processing, and delivery channels:

```
┌─────────────────────────────────────────────────────────────┐
│                     Delivery Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Mobile App   │  │ Telegram Bot │  │ Voice API    │     │
│  │(AWS Amplify) │  │  (Lambda)    │  │(Amazon Lex)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   API Gateway Layer                         │
│              (Amazon API Gateway)                           │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Cognito Auth │ Rate Limiting │ Request Routing      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Advisory Engine Core (AWS Lambda)              │
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
│                    AI/ML Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  SageMaker   │  │ Rekognition  │  │   Bedrock    │     │
│  │   Models     │  │ (Pest Det.)  │  │(Conv. AI)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  DynamoDB    │  │ ElastiCache  │  │      S3      │     │
│  │ (User Data)  │  │   (Redis)    │  │   (Images)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │  RDS (PG)    │  │  EventBridge │                        │
│  │(Structured)  │  │   (Events)   │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              External Integrations                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Weather    │  │    Market    │  │     SNS      │     │
│  │     API      │  │   Price API  │  │(Notifications)│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### AWS Architecture Details

**Serverless Compute Pattern**:
```
User Request → CloudFront → API Gateway → Lambda Function → DynamoDB/RDS
                                    ↓
                              EventBridge → SQS → Lambda (Async)
```

**ML Inference Pattern**:
```
Image Upload → S3 → Lambda Trigger → SageMaker Endpoint → DynamoDB
                                  ↓
                            Rekognition Custom Labels
```

**Real-time Notification Pattern**:
```
Risk Detection → EventBridge Rule → SNS Topic → SMS/Push
                                              ↓
                                         SQS → Lambda → Telegram API
```

**Data Sync Pattern (Offline-First)**:
```
Mobile App (Local DB) ←→ AppSync (GraphQL) ←→ DynamoDB
                              ↓
                        Conflict Resolution
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

**AWS Services Used**:
- AWS Lambda - Serverless compute for recommendation logic
- Amazon SageMaker - ML model hosting for crop prediction
- Amazon DynamoDB - Crop database and user profiles
- Amazon ElastiCache - Caching recommendation results

**Inputs**:
- `SoilProfile`: { soilType, pH, nitrogen, phosphorus, potassium, organicMatter, moisture }
- `SeasonalContext`: { season, month, region, rainfall }
- `LocationData`: { latitude, longitude, district, state }

**Outputs**:
- `CropRecommendation[]`: Array of recommended crops with scores

**AWS Lambda Function Structure**:
```python
import boto3
import json

# AWS SDK clients
dynamodb = boto3.resource('dynamodb')
sagemaker_runtime = boto3.client('sagemaker-runtime')
elasticache = boto3.client('elasticache')

def lambda_handler(event, context):
    """
    Lambda function for crop recommendation
    Triggered by API Gateway
    """
    # Parse input
    body = json.loads(event['body'])
    soil_profile = body['soil_profile']
    season = body['season']
    location = body['location']
    
    # Check cache first
    cache_key = f"crop_rec_{hash(json.dumps(body))}"
    cached_result = get_from_cache(cache_key)
    if cached_result:
        return {
            'statusCode': 200,
            'body': json.dumps(cached_result)
        }
    
    # Get crop database from DynamoDB
    crops_table = dynamodb.Table('CropDatabase')
    candidate_crops = crops_table.query(
        IndexName='RegionIndex',
        KeyConditionExpression='state = :state',
        ExpressionAttributeValues={':state': location['state']}
    )
    
    # Invoke SageMaker endpoint for ML-based scoring
    features = prepare_features(soil_profile, season, location)
    sagemaker_response = sagemaker_runtime.invoke_endpoint(
        EndpointName='crop-recommendation-endpoint',
        ContentType='application/json',
        Body=json.dumps(features)
    )
    
    predictions = json.loads(sagemaker_response['Body'].read())
    
    # Combine rule-based and ML scores
    recommendations = rank_crops(candidate_crops['Items'], predictions)
    
    # Cache result
    set_cache(cache_key, recommendations, ttl=3600)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'recommendations': recommendations[:5],
            'cached': False
        })
    }
```

**SageMaker Model Deployment**:
```python
import sagemaker
from sagemaker.sklearn import SKLearnModel

# Deploy trained model to SageMaker endpoint
sklearn_model = SKLearnModel(
    model_data='s3://crop-advisory-models/crop-recommender/model.tar.gz',
    role=sagemaker_role,
    entry_point='inference.py',
    framework_version='1.0-1'
)

predictor = sklearn_model.deploy(
    instance_type='ml.t2.medium',
    initial_instance_count=1,
    endpoint_name='crop-recommendation-endpoint'
)
```

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

**AWS Services Used**:
- Amazon Rekognition Custom Labels - Image classification for pest detection
- Amazon S3 - Image storage
- AWS Lambda - Image preprocessing and orchestration
- Amazon DynamoDB - Detection results storage

**Inputs**:
- `Image`: Binary image data (JPEG/PNG)
- `CropType`: Optional context for better accuracy

**Outputs**:
- `PestDetection`: { pestName, confidence, severity, treatments[] }

**AWS Architecture**:
```
User Upload → API Gateway → Lambda (Presigned URL) → S3
                                    ↓
                            S3 Event → Lambda (Process)
                                    ↓
                            Rekognition Custom Labels
                                    ↓
                            DynamoDB (Results) → SNS (Notify User)
```

**Lambda Function for Pest Detection**:
```python
import boto3
import json

s3 = boto3.client('s3')
rekognition = boto3.client('rekognition')
dynamodb = boto3.resource('dynamodb')

def lambda_handler(event, context):
    """
    Process uploaded image for pest detection
    Triggered by S3 event
    """
    # Get image from S3
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    # Validate image
    if not is_valid_image(bucket, key):
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid image format'})
        }
    
    # Call Rekognition Custom Labels
    response = rekognition.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:region:account:project/pest-detection/version/1',
        Image={'S3Object': {'Bucket': bucket, 'Name': key}},
        MinConfidence=60
    )
    
    # Process results
    if not response['CustomLabels']:
        return {
            'statusCode': 200,
            'body': json.dumps({
                'detected': False,
                'message': 'Unable to identify. Please upload clearer image.'
            })
        }
    
    top_prediction = response['CustomLabels'][0]
    
    # Get treatment recommendations from DynamoDB
    treatments_table = dynamodb.Table('PestTreatments')
    treatment_data = treatments_table.get_item(
        Key={'pest_name': top_prediction['Name']}
    )
    
    result = {
        'detected': True,
        'pest_name': top_prediction['Name'],
        'confidence': top_prediction['Confidence'] / 100,
        'severity': calculate_severity(top_prediction),
        'treatments': treatment_data['Item']['treatments'],
        'image_url': f"https://{bucket}.s3.amazonaws.com/{key}"
    }
    
    # Store result in DynamoDB
    results_table = dynamodb.Table('PestDetectionResults')
    results_table.put_item(Item={
        'detection_id': context.request_id,
        'image_key': key,
        **result,
        'timestamp': context.get_remaining_time_in_millis()
    })
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }

def calculate_severity(prediction):
    """Calculate severity based on confidence and pest type"""
    confidence = prediction['Confidence']
    pest_name = prediction['Name']
    
    # High confidence + destructive pest = critical
    if confidence > 90 and pest_name in ['Aphids', 'Caterpillar', 'Blight']:
        return 'critical'
    elif confidence > 75:
        return 'high'
    elif confidence > 60:
        return 'medium'
    else:
        return 'low'
```

**Rekognition Custom Labels Training**:
```python
import boto3

rekognition = boto3.client('rekognition')

# Create project
project_response = rekognition.create_project(
    ProjectName='pest-detection'
)

# Create dataset
dataset_response = rekognition.create_dataset(
    DatasetType='TRAIN',
    DatasetSource={
        'GroundTruthManifest': {
            'S3Object': {
                'Bucket': 'crop-advisory-training-data',
                'Name': 'pest-images/manifest.json'
            }
        }
    },
    ProjectArn=project_response['ProjectArn']
)

# Train model
training_response = rekognition.create_project_version(
    ProjectArn=project_response['ProjectArn'],
    VersionName='v1',
    OutputConfig={
        'S3Bucket': 'crop-advisory-models',
        'S3KeyPrefix': 'pest-detection/'
    },
    TrainingData={'Assets': [{'GroundTruthManifest': {...}}]},
    TestingData={'Assets': [{'GroundTruthManifest': {...}}]}
)

# Start model
rekognition.start_project_version(
    ProjectVersionArn=training_response['ProjectVersionArn'],
    MinInferenceUnits=1
)
```

**Alternative: SageMaker Custom Model**:
```python
# For more control, deploy custom CNN on SageMaker
from sagemaker.pytorch import PyTorchModel

pytorch_model = PyTorchModel(
    model_data='s3://crop-advisory-models/pest-detector/model.tar.gz',
    role=sagemaker_role,
    entry_point='inference.py',
    framework_version='1.12',
    py_version='py38'
)

predictor = pytorch_model.deploy(
    instance_type='ml.g4dn.xlarge',  # GPU instance for faster inference
    initial_instance_count=1,
    endpoint_name='pest-detection-endpoint'
)
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

**AWS Services Used**:
- Amazon SNS - Push notifications and SMS delivery
- Amazon SQS - Message queuing for reliable delivery
- AWS Lambda - Notification processing logic
- Amazon DynamoDB - User preferences and notification history
- Amazon EventBridge - Event-driven notification triggers

**Inputs**:
- `Alert`: { type, severity, message, targetUsers[] }
- `UserPreferences`: { channels[], quietHours, maxDailyAlerts }

**Outputs**:
- `DeliveryStatus`: { sent, failed, queued }

**AWS Architecture**:
```
Event Source → EventBridge Rule → Lambda (Process)
                                       ↓
                              Check Preferences (DynamoDB)
                                       ↓
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
                SNS (Push)         SNS (SMS)         SQS (Telegram)
                    ↓                  ↓                  ↓
              Mobile Device        Phone Number      Lambda → Telegram API
```

**Lambda Function for Notification Processing**:
```python
import boto3
import json
from datetime import datetime

sns = boto3.client('sns')
sqs = boto3.client('sqs')
dynamodb = boto3.resource('dynamodb')
eventbridge = boto3.client('events')

def lambda_handler(event, context):
    """
    Process and route notifications based on user preferences
    Triggered by EventBridge
    """
    alert = json.loads(event['detail'])
    user_id = alert['user_id']
    
    # Get user preferences
    users_table = dynamodb.Table('UserProfiles')
    user = users_table.get_item(Key={'user_id': user_id})['Item']
    preferences = user['notification_preferences']
    
    # Check alert fatigue limits
    if not should_send_alert(user_id, alert['severity'], preferences):
        return {
            'statusCode': 200,
            'body': json.dumps({'status': 'queued', 'reason': 'rate_limited'})
        }
    
    # Determine delivery channels
    channels = select_channels(preferences, alert['severity'])
    
    results = []
    
    # Send via SNS (Push notifications)
    if 'push' in channels and user.get('device_token'):
        push_result = send_push_notification(user['device_token'], alert)
        results.append(push_result)
    
    # Send via SNS (SMS)
    if 'sms' in channels and user.get('phone_number'):
        sms_result = send_sms(user['phone_number'], alert)
        results.append(sms_result)
    
    # Queue for Telegram
    if 'telegram' in channels and user.get('telegram_id'):
        telegram_result = queue_telegram_message(user['telegram_id'], alert)
        results.append(telegram_result)
    
    # Log notification
    log_notification(user_id, alert, results)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'results': results})
    }

def send_push_notification(device_token, alert):
    """Send push notification via SNS"""
    try:
        # Create SNS platform endpoint if not exists
        endpoint_arn = get_or_create_endpoint(device_token)
        
        # Publish to endpoint
        response = sns.publish(
            TargetArn=endpoint_arn,
            Message=json.dumps({
                'default': alert['message'],
                'GCM': json.dumps({
                    'notification': {
                        'title': alert['title'],
                        'body': alert['message'],
                        'priority': 'high' if alert['severity'] in ['high', 'critical'] else 'normal'
                    },
                    'data': alert
                })
            }),
            MessageStructure='json'
        )
        
        return {'channel': 'push', 'status': 'sent', 'message_id': response['MessageId']}
    except Exception as e:
        return {'channel': 'push', 'status': 'failed', 'error': str(e)}

def send_sms(phone_number, alert):
    """Send SMS via SNS"""
    try:
        response = sns.publish(
            PhoneNumber=phone_number,
            Message=f"{alert['title']}\n\n{alert['message']}",
            MessageAttributes={
                'AWS.SNS.SMS.SMSType': {
                    'DataType': 'String',
                    'StringValue': 'Transactional'  # For critical alerts
                }
            }
        )
        
        return {'channel': 'sms', 'status': 'sent', 'message_id': response['MessageId']}
    except Exception as e:
        return {'channel': 'sms', 'status': 'failed', 'error': str(e)}

def queue_telegram_message(telegram_id, alert):
    """Queue message for Telegram delivery"""
    try:
        response = sqs.send_message(
            QueueUrl='https://sqs.region.amazonaws.com/account/telegram-notifications',
            MessageBody=json.dumps({
                'telegram_id': telegram_id,
                'alert': alert
            }),
            MessageAttributes={
                'Priority': {
                    'StringValue': alert['severity'],
                    'DataType': 'String'
                }
            }
        )
        
        return {'channel': 'telegram', 'status': 'queued', 'message_id': response['MessageId']}
    except Exception as e:
        return {'channel': 'telegram', 'status': 'failed', 'error': str(e)}

def should_send_alert(user_id, severity, preferences):
    """Check if alert should be sent based on rate limits and quiet hours"""
    # Critical alerts always sent
    if severity == 'critical':
        return True
    
    # Check quiet hours
    now = datetime.now().time()
    quiet_start = datetime.strptime(preferences['quiet_hours']['start'], '%H:%M').time()
    quiet_end = datetime.strptime(preferences['quiet_hours']['end'], '%H:%M').time()
    
    if quiet_start <= now <= quiet_end:
        return False
    
    # Check daily limit
    notifications_table = dynamodb.Table('NotificationHistory')
    today_count = notifications_table.query(
        KeyConditionExpression='user_id = :uid AND sent_at > :today',
        ExpressionAttributeValues={
            ':uid': user_id,
            ':today': datetime.now().replace(hour=0, minute=0, second=0).isoformat()
        }
    )['Count']
    
    return today_count < preferences.get('max_daily_alerts', 5)

def select_channels(preferences, severity):
    """Select notification channels based on severity"""
    if severity == 'critical':
        return ['push', 'sms', 'telegram']  # All channels
    elif severity == 'high':
        return preferences.get('primary_channels', ['push', 'telegram'])
    else:
        return [preferences.get('default_channel', 'push')]
```

**EventBridge Rules for Automatic Notifications**:
```python
# Create EventBridge rule for weather alerts
eventbridge.put_rule(
    Name='weather-alert-rule',
    EventPattern=json.dumps({
        'source': ['crop.advisory.weather'],
        'detail-type': ['Severe Weather Detected'],
        'detail': {
            'severity': ['high', 'critical']
        }
    }),
    State='ENABLED'
)

# Add Lambda target
eventbridge.put_targets(
    Rule='weather-alert-rule',
    Targets=[{
        'Id': '1',
        'Arn': 'arn:aws:lambda:region:account:function:notification-processor'
    }]
)
```

### 8. Voice Interface (Post-MVP)

**Purpose**: Enable voice-based interaction for users with limited literacy.

**AWS Services Used**:
- Amazon Transcribe - Speech-to-text conversion
- Amazon Polly - Text-to-speech synthesis
- Amazon Lex - Conversational bot framework
- Amazon Bedrock - Natural language understanding
- AWS Lambda - Voice processing orchestration
- Amazon S3 - Audio file storage

**Inputs**:
- `AudioData`: Voice recording
- `Language`: User's preferred language

**Outputs**:
- `TranscribedText`: Recognized text
- `SpokenResponse`: Audio response

**AWS Architecture**:
```
Voice Input → API Gateway → Lambda → S3 (Audio Storage)
                                ↓
                        Amazon Transcribe
                                ↓
                        Amazon Lex (Intent Detection)
                                ↓
                        Advisory Engine (Lambda)
                                ↓
                        Amazon Polly (TTS)
                                ↓
                        S3 (Response Audio) → CloudFront → User
```

**Lambda Function for Voice Processing**:
```python
import boto3
import json

transcribe = boto3.client('transcribe')
polly = boto3.client('polly')
lex = boto3.client('lexv2-runtime')
bedrock = boto3.client('bedrock-runtime')
s3 = boto3.client('s3')

def lambda_handler(event, context):
    """
    Process voice query end-to-end
    """
    # Get audio file from S3
    audio_key = event['audio_key']
    language = event.get('language', 'hi-IN')  # Default to Hindi
    
    # Transcribe audio
    transcription = transcribe_audio(audio_key, language)
    
    # Process through Lex for intent detection
    lex_response = lex.recognize_text(
        botId='crop-advisory-bot',
        botAliasId='PROD',
        localeId=language,
        sessionId=event['user_id'],
        text=transcription
    )
    
    # Get response from advisory engine
    if lex_response['sessionState']['intent']['name'] == 'CropRecommendation':
        response_text = get_crop_recommendations(event['user_id'])
    elif lex_response['sessionState']['intent']['name'] == 'WeatherQuery':
        response_text = get_weather_info(event['user_id'])
    else:
        # Use Bedrock for general queries
        response_text = get_bedrock_response(transcription, event['user_id'])
    
    # Convert response to speech
    audio_url = text_to_speech(response_text, language)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'transcription': transcription,
            'response_text': response_text,
            'response_audio_url': audio_url
        })
    }

def transcribe_audio(audio_key, language):
    """Transcribe audio using Amazon Transcribe"""
    job_name = f"transcribe-{audio_key.replace('/', '-')}"
    
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': f's3://crop-advisory-audio/{audio_key}'},
        MediaFormat='mp3',
        LanguageCode=language,
        Settings={
            'ShowSpeakerLabels': False,
            'MaxSpeakerLabels': 1
        }
    )
    
    # Wait for completion
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(2)
    
    # Get transcript
    transcript_uri = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
    transcript_data = requests.get(transcript_uri).json()
    
    return transcript_data['results']['transcripts'][0]['transcript']

def text_to_speech(text, language):
    """Convert text to speech using Amazon Polly"""
    # Map language codes to Polly voice IDs
    voice_map = {
        'hi-IN': 'Aditi',  # Hindi
        'en-IN': 'Raveena',  # English (Indian)
        'ta-IN': 'Kajal',  # Tamil (if available)
        'te-IN': 'Kajal',  # Telugu
    }
    
    voice_id = voice_map.get(language, 'Aditi')
    
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat='mp3',
        VoiceId=voice_id,
        Engine='neural',  # Better quality
        LanguageCode=language
    )
    
    # Save to S3
    audio_key = f"responses/{context.request_id}.mp3"
    s3.put_object(
        Bucket='crop-advisory-audio',
        Key=audio_key,
        Body=response['AudioStream'].read(),
        ContentType='audio/mpeg'
    )
    
    # Return CloudFront URL
    return f"https://d1234567890.cloudfront.net/{audio_key}"

def get_bedrock_response(query, user_id):
    """Use Amazon Bedrock for conversational AI"""
    # Get user context
    user_context = get_user_context(user_id)
    
    prompt = f"""You are an agricultural advisor for Indian farmers.
    
User context: {json.dumps(user_context)}
Farmer question: {query}

Provide a helpful, concise response in simple language suitable for farmers."""
    
    response = bedrock.invoke_model(
        modelId='anthropic.claude-v2',
        body=json.dumps({
            'prompt': prompt,
            'max_tokens_to_sample': 300,
            'temperature': 0.7
        })
    )
    
    result = json.loads(response['body'].read())
    return result['completion']
```

**Amazon Lex Bot Configuration**:
```python
# Create Lex bot for intent detection
lex_client = boto3.client('lexv2-models')

bot_response = lex_client.create_bot(
    botName='CropAdvisoryBot',
    description='Agricultural advisory bot for farmers',
    roleArn='arn:aws:iam::account:role/LexBotRole',
    dataPrivacy={'childDirected': False},
    idleSessionTTLInSeconds=300
)

# Add intents
intents = [
    {
        'intentName': 'CropRecommendation',
        'sampleUtterances': [
            {'utterance': 'What crop should I plant'},
            {'utterance': 'Recommend crops for my farm'},
            {'utterance': 'Which crop is best for this season'}
        ]
    },
    {
        'intentName': 'WeatherQuery',
        'sampleUtterances': [
            {'utterance': 'What is the weather'},
            {'utterance': 'Will it rain today'},
            {'utterance': 'Weather forecast'}
        ]
    },
    {
        'intentName': 'IrrigationAdvice',
        'sampleUtterances': [
            {'utterance': 'Should I water my crops'},
            {'utterance': 'When to irrigate'},
            {'utterance': 'Irrigation schedule'}
        ]
    }
]

for intent in intents:
    lex_client.create_intent(
        botId=bot_response['botId'],
        botVersion='DRAFT',
        localeId='hi_IN',
        **intent
    )
```

### 9. Telegram Bot Interface

**Purpose**: Provide full system access through Telegram for users without smartphones.

**AWS Services Used**:
- AWS Lambda - Bot logic and message handling
- Amazon API Gateway - Webhook endpoint for Telegram
- Amazon DynamoDB - User sessions and state management
- Amazon S3 - Image storage from Telegram
- Amazon SQS - Message queue for async processing

**Commands**:
- `/start` - Register and set preferences
- `/recommend` - Get crop recommendations
- `/weather` - Get weather and irrigation advice
- `/calendar` - View crop calendar
- `/market` - Check market prices
- `/help` - Get help and usage instructions

**Image Handling**:
- User sends image → Bot processes through Pest Detector → Returns identification and treatment

**AWS Architecture**:
```
Telegram → API Gateway (Webhook) → Lambda (Bot Handler)
                                        ↓
                                   DynamoDB (Sessions)
                                        ↓
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            Advisory Engine      Rekognition         S3 (Images)
                    ↓                   ↓                   
            Response → Lambda → Telegram API
```

**Lambda Function for Telegram Bot**:
```python
import boto3
import json
import requests
from telegram import Bot, Update
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters

dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')

# Telegram bot token from Secrets Manager
secrets = boto3.client('secretsmanager')
bot_token = json.loads(
    secrets.get_secret_value(SecretId='telegram-bot-token')['SecretString']
)['token']

bot = Bot(token=bot_token)

def lambda_handler(event, context):
    """
    Handle Telegram webhook updates
    Triggered by API Gateway
    """
    # Parse Telegram update
    update = Update.de_json(json.loads(event['body']), bot)
    
    # Get or create user session
    user_id = str(update.effective_user.id)
    session = get_or_create_session(user_id)
    
    # Handle different message types
    if update.message.text:
        if update.message.text.startswith('/'):
            response = handle_command(update.message.text, session)
        else:
            response = handle_text_query(update.message.text, session)
    
    elif update.message.voice:
        response = handle_voice_message(update.message.voice, session)
    
    elif update.message.photo:
        response = handle_photo(update.message.photo[-1], session)
    
    else:
        response = "Sorry, I can only process text, voice, and images."
    
    # Send response
    bot.send_message(
        chat_id=update.effective_chat.id,
        text=response,
        parse_mode='Markdown'
    )
    
    return {'statusCode': 200}

def handle_command(command, session):
    """Handle bot commands"""
    user_id = session['user_id']
    
    if command == '/start':
        return register_user(user_id)
    
    elif command == '/recommend':
        # Invoke crop recommendation Lambda
        response = lambda_client.invoke(
            FunctionName='crop-recommender',
            InvocationType='RequestResponse',
            Payload=json.dumps({
                'user_id': user_id,
                'source': 'telegram'
            })
        )
        
        recommendations = json.loads(response['Payload'].read())
        return format_crop_recommendations(recommendations)
    
    elif command == '/weather':
        response = lambda_client.invoke(
            FunctionName='weather-advisor',
            Payload=json.dumps({'user_id': user_id})
        )
        
        weather_data = json.loads(response['Payload'].read())
        return format_weather_info(weather_data)
    
    elif command == '/calendar':
        response = lambda_client.invoke(
            FunctionName='calendar-manager',
            Payload=json.dumps({'user_id': user_id})
        )
        
        calendar = json.loads(response['Payload'].read())
        return format_calendar(calendar)
    
    elif command == '/market':
        response = lambda_client.invoke(
            FunctionName='market-insights',
            Payload=json.dumps({'user_id': user_id})
        )
        
        market_data = json.loads(response['Payload'].read())
        return format_market_prices(market_data)
    
    elif command == '/help':
        return """
*Available Commands:*

/recommend - Get crop recommendations
/weather - Check weather and irrigation advice
/calendar - View your crop calendar
/market - Check market prices
/help - Show this help message

You can also:
• Send a photo of a plant leaf for pest detection
• Send a voice message with your question
• Type your question in text
"""
    
    else:
        return "Unknown command. Type /help for available commands."

def handle_photo(photo, session):
    """Handle image for pest detection"""
    user_id = session['user_id']
    
    # Download image from Telegram
    file = bot.get_file(photo.file_id)
    image_data = requests.get(file.file_path).content
    
    # Upload to S3
    image_key = f"telegram-uploads/{user_id}/{photo.file_id}.jpg"
    s3.put_object(
        Bucket='crop-advisory-images',
        Key=image_key,
        Body=image_data,
        ContentType='image/jpeg'
    )
    
    # Invoke pest detection
    response = lambda_client.invoke(
        FunctionName='pest-detector',
        Payload=json.dumps({
            'image_bucket': 'crop-advisory-images',
            'image_key': image_key,
            'user_id': user_id
        })
    )
    
    result = json.loads(response['Payload'].read())
    
    if result['detected']:
        return f"""
*Pest Detection Result*

🐛 *Identified:* {result['pest_name']}
📊 *Confidence:* {result['confidence']*100:.1f}%
⚠️ *Severity:* {result['severity'].upper()}

*Recommended Treatments:*
{format_treatments(result['treatments'])}

*Preventive Measures:*
{format_preventive_measures(result['treatments'])}
"""
    else:
        return "Unable to identify the issue. Please upload a clearer image of the affected leaf."

def handle_voice_message(voice, session):
    """Handle voice message"""
    user_id = session['user_id']
    
    # Download voice file
    file = bot.get_file(voice.file_id)
    audio_data = requests.get(file.file_path).content
    
    # Upload to S3
    audio_key = f"telegram-voice/{user_id}/{voice.file_id}.ogg"
    s3.put_object(
        Bucket='crop-advisory-audio',
        Key=audio_key,
        Body=audio_data,
        ContentType='audio/ogg'
    )
    
    # Invoke voice processor
    response = lambda_client.invoke(
        FunctionName='voice-processor',
        Payload=json.dumps({
            'audio_key': audio_key,
            'user_id': user_id,
            'language': session.get('language', 'hi-IN')
        })
    )
    
    result = json.loads(response['Payload'].read())
    
    # Send text response (Telegram will handle TTS on client side)
    return result['response_text']

def handle_text_query(text, session):
    """Handle natural language text query"""
    user_id = session['user_id']
    
    # Use Bedrock for conversational understanding
    bedrock = boto3.client('bedrock-runtime')
    
    user_context = get_user_context(user_id)
    
    prompt = f"""You are an agricultural advisor for Indian farmers.

User context: {json.dumps(user_context)}
Farmer question: {text}

Provide a helpful, concise response in simple language."""
    
    response = bedrock.invoke_model(
        modelId='anthropic.claude-v2',
        body=json.dumps({
            'prompt': prompt,
            'max_tokens_to_sample': 300
        })
    )
    
    result = json.loads(response['body'].read())
    return result['completion']

def get_or_create_session(user_id):
    """Get or create user session in DynamoDB"""
    sessions_table = dynamodb.Table('TelegramSessions')
    
    response = sessions_table.get_item(Key={'user_id': user_id})
    
    if 'Item' in response:
        return response['Item']
    else:
        # Create new session
        session = {
            'user_id': user_id,
            'language': 'hi-IN',
            'created_at': datetime.now().isoformat()
        }
        sessions_table.put_item(Item=session)
        return session

def format_crop_recommendations(recommendations):
    """Format crop recommendations for Telegram"""
    text = "*🌾 Crop Recommendations*\n\n"
    
    for i, crop in enumerate(recommendations['recommendations'], 1):
        text += f"{i}. *{crop['crop_name']}* ({crop['confidence']*100:.0f}% match)\n"
        text += f"   • Yield: {crop['expected_yield']['min']}-{crop['expected_yield']['max']} {crop['expected_yield']['unit']}\n"
        text += f"   • Duration: {crop['growing_duration']} days\n"
        text += f"   • Water: {crop['water_requirement']}\n"
        text += f"   • Market: {crop['market_demand']}\n\n"
    
    return text
```

**API Gateway Webhook Setup**:
```python
# Set Telegram webhook to API Gateway endpoint
import requests

api_gateway_url = "https://abc123.execute-api.region.amazonaws.com/prod/telegram-webhook"

telegram_api_url = f"https://api.telegram.org/bot{bot_token}/setWebhook"
response = requests.post(telegram_api_url, json={
    'url': api_gateway_url,
    'allowed_updates': ['message', 'callback_query']
})

print(response.json())
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

## ML & EDA-Ready Schema for Small and Informal Producers

This comprehensive schema extends the core data models to support machine learning, exploratory data analysis, and multi-activity agricultural production (crops, fishery, poultry, beekeeping, livestock, insect farming). The schema is designed for progressive data collection, allowing minimal required fields initially while supporting rich analytics as more data becomes available.

### Design Principles for ML/EDA Schema

1. **Progressive Disclosure**: Minimal required fields for onboarding, optional fields for enhanced analytics
2. **Multi-Activity Support**: Unified schema for diverse agricultural activities
3. **Derived Features**: Automatic computation of ML features from raw data
4. **Time-Series Ready**: Event logging for temporal analysis
5. **Privacy-Preserving**: Anonymization support for analytics
6. **Offline-First**: Local computation of derived features

### 1. Producer (Root Entity)

```typescript
interface Producer {
    // Minimal Required
    id: UUID;
    name: string;
    phone: string;
    village: string;
    district: string;
    state: string;
    
    // Optional / Progressive
    gender?: "male" | "female" | "other" | "prefer_not_to_say";
    age_range?: "18-25" | "25-40" | "40-60" | "60+";
    education_level?: "no_formal" | "primary" | "secondary" | "higher_secondary" | "graduate";
    
    preferred_language?: "hi" | "en" | "ta" | "te" | "kn" | "mr" | "bn" | "gu";
    primary_activity_type: "crop" | "fishery" | "poultry" | "beekeeping" | "livestock" | "insect" | "mixed";
    
    years_of_experience?: number;
    land_ownership_type?: "owned" | "leased" | "community" | "sharecropping";
    irrigation_access?: "yes" | "no" | "seasonal";
    
    credit_access?: boolean;
    bank_account_available?: boolean;
    
    smartphone_type?: "android_basic" | "android_advanced" | "feature_phone" | "no_phone";
    
    // ML / EDA Derived Features [Optional / Auto-computed]
    risk_appetite_score?: number; // 0-1, derived from activity history
    avg_income_per_year?: number; // derived from production outcomes
    diversification_score?: number; // 0-1, multi-activity diversity index
    seasonal_activity_patterns?: Record<string, number>; // activity distribution by month
    
    created_at: Date;
    last_active_at: Date;
}
```

**Derived Feature Computation Examples**:
```typescript
// Diversification Score: Shannon entropy of activity types
function computeDiversificationScore(producer: Producer): number {
    const activities = getProducerActivities(producer.id);
    const typeCounts = countBy(activities, 'activity_category');
    const total = activities.length;
    
    let entropy = 0;
    for (const count of Object.values(typeCounts)) {
        const p = count / total;
        entropy -= p * Math.log2(p);
    }
    
    // Normalize to 0-1 scale
    const maxEntropy = Math.log2(Object.keys(typeCounts).length);
    return entropy / maxEntropy;
}

// Risk Appetite Score: Based on adoption of new practices, investment patterns
function computeRiskAppetiteScore(producer: Producer): number {
    const activities = getProducerActivities(producer.id);
    
    const newCropVarieties = activities.filter(a => a.is_new_variety).length;
    const highInvestmentActivities = activities.filter(a => a.initial_investment > threshold).length;
    const experimentalPractices = activities.filter(a => a.is_experimental).length;
    
    return normalize([newCropVarieties, highInvestmentActivities, experimentalPractices]);
}
```

### 2. ProductionUnit (Generic Multi-Activity Unit)

```typescript
interface ProductionUnit {
    id: UUID;
    producer_id: UUID;
    
    name: string;
    unit_type: "land_plot" | "fish_pond" | "apiary" | "poultry_shed" | "insect_unit" | "livestock_pen";
    
    // Basic size
    area_size?: number;
    area_unit?: "acre" | "hectare" | "sqft" | "pond_sq_m" | "shed_sq_ft";
    
    geo_latitude?: number;
    geo_longitude?: number;
    
    // Optional / Inferred
    elevation_meters?: number; // GPS or DEM [Optional / Auto]
    slope_percentage?: number; // derived from geo [Optional / Auto]
    soil_type_simple?: "clay" | "sandy" | "loamy" | "silt" | "red" | "black" | "alluvial";
    
    water_source?: "borewell" | "canal" | "river" | "pond" | "rainwater" | "municipal";
    infrastructure_type?: "open_field" | "greenhouse" | "polyhouse" | "shed" | "pond" | "cage";
    
    // ML / EDA Derived Features [Optional / Auto]
    ndvi_index?: number; // satellite vegetation index (0-1)
    water_availability_score?: number; // sensor or remote sensed (0-1)
    flood_risk_score?: number; // derived from past events / geo (0-1)
    drought_risk_score?: number; // derived from past weather (0-1)
    
    created_at: Date;
}
```

**Satellite Data Integration**:
```typescript
// NDVI computation from Sentinel-2 or Landsat imagery
async function computeNDVI(unit: ProductionUnit): Promise<number> {
    if (!unit.geo_latitude || !unit.geo_longitude) return null;
    
    const imagery = await fetchSatelliteImagery({
        lat: unit.geo_latitude,
        lon: unit.geo_longitude,
        date: new Date(),
        bands: ['red', 'nir'] // Near-infrared and red bands
    });
    
    const ndvi = (imagery.nir - imagery.red) / (imagery.nir + imagery.red);
    return ndvi;
}
```

### 3. Activity (Cycle / Batch / Production Event)

```typescript
interface Activity {
    id: UUID;
    production_unit_id: UUID;
    
    activity_category: "crop" | "fish_batch" | "poultry_batch" | "bee_cycle" | "insect_batch" | "livestock_cycle";
    name: string; // wheat | rohu | broiler | honeybee | black_soldier_fly | dairy_cow
    variety_or_species?: string;
    
    start_date: Date;
    expected_end_date?: Date;
    actual_end_date?: Date;
    
    status: "active" | "completed" | "failed" | "abandoned";
    
    scale_count?: number; // number of chicks | fish count | hives | livestock count
    initial_investment_estimate?: number;
    
    notes_voice_input?: string;
    
    // ML / EDA Derived Features [Optional / Auto]
    avg_growth_rate?: number; // derived from events / sensors
    mortality_rate?: number; // derived from IssueReports
    resource_efficiency_score?: number; // output / input ratio
    water_productivity?: number; // kg output per m³ water
    climate_sensitivity_index?: number; // derived from weather + outcome correlation
    revenue_per_unit?: number; // derived from outcome
    
    created_at: Date;
}
```

**Growth Rate Computation**:
```typescript
function computeAvgGrowthRate(activity: Activity): number {
    const events = getActivityEvents(activity.id, 'measurement');
    
    if (events.length < 2) return null;
    
    const growthRates = [];
    for (let i = 1; i < events.length; i++) {
        const timeDiff = daysBetween(events[i-1].event_timestamp, events[i].event_timestamp);
        const valueDiff = events[i].quantity_value - events[i-1].quantity_value;
        growthRates.push(valueDiff / timeDiff);
    }
    
    return average(growthRates);
}
```

### 4. ActivityEvent (Time-Series / Event Logging)

```typescript
interface ActivityEvent {
    id: UUID;
    activity_id: UUID;
    
    event_type: "sowing" | "feeding" | "vaccination" | "irrigation" | "harvest" | "sale" | "mortality" | "measurement" | "treatment";
    event_timestamp: Date;
    
    quantity_value?: number;
    quantity_unit?: string;
    
    cost_incurred?: number;
    
    photo_url?: string;
    voice_note_transcript?: string;
    
    weather_context_auto?: WeatherSnapshot; // linked weather data
    
    created_by: "farmer" | "system" | "sensor";
    
    // ML / EDA Derived Features [Optional / Auto]
    soil_moisture?: number; // sensor input (0-100%)
    water_temperature?: number; // fish pond (°C)
    feed_quality_score?: number; // derived or sensor (0-1)
    pest_pressure_index?: number; // derived from AI detection (0-1)
    intervention_effectiveness_score?: number; // derived from outcomes (0-1)
}
```

**Event Pattern Analysis**:
```typescript
// Detect intervention effectiveness by comparing outcomes before/after
function computeInterventionEffectiveness(event: ActivityEvent): number {
    if (event.event_type !== 'treatment' && event.event_type !== 'vaccination') return null;
    
    const activity = getActivity(event.activity_id);
    const beforeEvents = getActivityEvents(activity.id, 'issue', {
        before: event.event_timestamp,
        days: 7
    });
    const afterEvents = getActivityEvents(activity.id, 'issue', {
        after: event.event_timestamp,
        days: 7
    });
    
    const issueReductionRate = (beforeEvents.length - afterEvents.length) / beforeEvents.length;
    return Math.max(0, Math.min(1, issueReductionRate));
}
```

### 5. WeatherSnapshot

```typescript
interface WeatherSnapshot {
    snapshot_id: UUID;
    
    timestamp: Date;
    temperature_celsius?: number;
    min_temperature_celsius?: number;
    max_temperature_celsius?: number;
    
    rainfall_mm?: number;
    humidity_percentage?: number;
    wind_speed_m_s?: number;
    solar_radiation_w_m2?: number; // if sensor available
    dew_point_celsius?: number;
    
    weather_source: "api" | "station" | "sensor" | "manual";
    
    // ML / EDA Derived Features [Optional / Auto]
    growing_degree_days?: number; // accumulated heat units
    heatwave_flag?: boolean;
    coldwave_flag?: boolean;
    extreme_rainfall_flag?: boolean;
    drought_flag?: boolean;
}
```

**Growing Degree Days (GDD) Computation**:
```typescript
function computeGrowingDegreeDays(weather: WeatherSnapshot, baseTemp: number = 10): number {
    const avgTemp = (weather.min_temperature_celsius + weather.max_temperature_celsius) / 2;
    return Math.max(0, avgTemp - baseTemp);
}

// Accumulated GDD for crop growth stage prediction
function accumulatedGDD(activity: Activity): number {
    const weatherSnapshots = getWeatherForActivity(activity.id);
    return weatherSnapshots.reduce((sum, w) => sum + (w.growing_degree_days || 0), 0);
}
```

### 6. IssueReport (Generic Pest / Disease / Mortality / Water Quality)

```typescript
interface IssueReport {
    id: UUID;
    activity_id: UUID;
    detected_at: Date;
    
    issue_type: "pest" | "disease" | "mortality" | "water_quality" | "nutrient_deficiency" | "stress";
    image_url?: string;
    
    // ML / EDA Derived Features [Optional / Auto]
    ai_prediction?: string;
    confidence_score?: number; // 0-1
    severity_level?: "low" | "medium" | "high" | "critical";
    affected_percentage?: number; // affected area / stock (0-100%)
    
    farmer_action_taken?: string;
    cost_of_treatment?: number;
    
    outcome_impact_score?: number; // derived from subsequent ProductionOutcome (0-1)
}
```

### 7. ProductionOutcome

```typescript
interface ProductionOutcome {
    id: UUID;
    activity_id: UUID;
    
    total_output_quantity?: number;
    output_unit?: string; // kg | quintal | liters | eggs | honey_kg | fish_count
    
    total_revenue?: number;
    estimated_loss?: number;
    
    quality_grade?: "premium" | "grade_a" | "grade_b" | "grade_c" | "reject";
    sold_to?: "local_trader" | "mandi" | "direct_consumer" | "cooperative" | "processor";
    price_per_unit?: number;
    
    // ML / EDA Derived Features [Optional / Auto]
    yield_per_unit_area?: number; // derived (output / area)
    profit_margin?: number; // derived ((revenue - costs) / revenue)
    income_volatility_score?: number; // multi-season derived (std dev of profits)
    productivity_trend_score?: number; // historical trend derived (-1 to 1)
    
    created_at: Date;
}
```

**Productivity Trend Analysis**:
```typescript
function computeProductivityTrend(producer: Producer): number {
    const outcomes = getProducerOutcomes(producer.id, { last_n_seasons: 4 });
    
    if (outcomes.length < 2) return 0;
    
    const yields = outcomes.map(o => o.yield_per_unit_area);
    const trend = linearRegressionSlope(yields);
    
    // Normalize to -1 (declining) to 1 (improving)
    return Math.max(-1, Math.min(1, trend / Math.abs(average(yields))));
}
```

### 8. ResourceUsage (Water / Feed / Fertilizer / Labor / Energy)

```typescript
interface ResourceUsage {
    id: UUID;
    activity_id: UUID;
    
    resource_type: "water" | "feed" | "fertilizer" | "pesticide" | "electricity" | "labor" | "seed" | "medicine";
    quantity?: number;
    unit?: string;
    cost?: number;
    recorded_at: Date;
    
    // ML / EDA Derived Features [Optional / Auto]
    efficiency_score?: number; // output/input ratio (0-1)
    wastage_percentage?: number; // derived (0-100%)
}
```

### 9. RecommendationLog (Advice / System Guidance)

```typescript
interface RecommendationLog {
    id: UUID;
    producer_id: UUID;
    activity_id?: UUID;
    
    recommendation_type: "crop_selection" | "irrigation" | "pest_control" | "market_timing" | "risk_mitigation";
    message_text: string;
    generated_at: Date;
    
    // ML / EDA Derived Features [Optional / Auto]
    farmer_feedback?: "helpful" | "ignored" | "not_relevant" | "followed";
    followed?: boolean;
    impact_score?: number; // derived from outcome / events (0-1)
}
```

**Recommendation Impact Analysis**:
```typescript
function computeRecommendationImpact(recommendation: RecommendationLog): number {
    if (!recommendation.followed) return 0;
    
    const activity = getActivity(recommendation.activity_id);
    const outcome = getProductionOutcome(activity.id);
    
    // Compare with similar activities that didn't follow recommendation
    const similarActivities = findSimilarActivities(activity, { followed_recommendation: false });
    const avgYieldWithout = average(similarActivities.map(a => a.outcome.yield_per_unit_area));
    
    const improvementRatio = outcome.yield_per_unit_area / avgYieldWithout;
    return Math.max(0, Math.min(1, (improvementRatio - 1) / 0.5)); // Normalize to 0-1
}
```

### 10. DerivedFeature (Feature Store Table)

```typescript
interface DerivedFeature {
    entity_type: "producer" | "activity" | "unit" | "region";
    entity_id: UUID;
    feature_name: string;
    feature_value: number;
    computed_at: Date;
    
    // Optional metadata
    computation_method?: string;
    confidence?: number;
    data_sources?: string[];
}
```

**Feature Store Usage**:
```typescript
// Centralized feature computation and caching
class FeatureStore {
    async getFeature(entityType: string, entityId: UUID, featureName: string): Promise<number> {
        // Check cache first
        const cached = await this.getCachedFeature(entityType, entityId, featureName);
        if (cached && !this.isStale(cached)) {
            return cached.feature_value;
        }
        
        // Compute feature
        const value = await this.computeFeature(entityType, entityId, featureName);
        
        // Store in feature store
        await this.storeFeature({
            entity_type: entityType,
            entity_id: entityId,
            feature_name: featureName,
            feature_value: value,
            computed_at: new Date()
        });
        
        return value;
    }
    
    private async computeFeature(entityType: string, entityId: UUID, featureName: string): Promise<number> {
        const computeFn = this.featureRegistry[featureName];
        if (!computeFn) throw new Error(`Unknown feature: ${featureName}`);
        
        return await computeFn(entityType, entityId);
    }
}
```

### ML Pipeline Integration

**Feature Engineering for Crop Recommendation Model**:
```typescript
async function extractFeaturesForCropRecommendation(producer: Producer, unit: ProductionUnit): Promise<FeatureVector> {
    const featureStore = new FeatureStore();
    
    return {
        // Producer features
        years_experience: producer.years_of_experience || 0,
        risk_appetite: await featureStore.getFeature('producer', producer.id, 'risk_appetite_score'),
        diversification: await featureStore.getFeature('producer', producer.id, 'diversification_score'),
        
        // Unit features
        soil_type_encoded: encodeSoilType(unit.soil_type_simple),
        elevation: unit.elevation_meters || 0,
        ndvi: await featureStore.getFeature('unit', unit.id, 'ndvi_index'),
        water_availability: await featureStore.getFeature('unit', unit.id, 'water_availability_score'),
        
        // Weather features
        avg_rainfall_30d: await getAvgRainfall(unit, 30),
        avg_temp_30d: await getAvgTemperature(unit, 30),
        gdd_accumulated: await getAccumulatedGDD(unit),
        
        // Historical performance
        past_yield_avg: await getAvgYield(producer.id, unit.id),
        past_profit_margin: await getAvgProfitMargin(producer.id, unit.id),
        
        // Market features
        market_demand_index: await getMarketDemandIndex(unit.district),
        price_volatility: await getPriceVolatility(unit.district)
    };
}
```

### EDA-Ready Data Export

```typescript
// Export data in formats suitable for exploratory data analysis
async function exportForEDA(filters: ExportFilters): Promise<DataFrame> {
    const data = await db.query(`
        SELECT 
            p.id as producer_id,
            p.district,
            p.state,
            p.primary_activity_type,
            p.years_of_experience,
            p.diversification_score,
            
            u.unit_type,
            u.area_size,
            u.soil_type_simple,
            u.ndvi_index,
            
            a.activity_category,
            a.name as crop_name,
            a.start_date,
            a.actual_end_date,
            a.avg_growth_rate,
            a.mortality_rate,
            
            o.total_output_quantity,
            o.yield_per_unit_area,
            o.total_revenue,
            o.profit_margin,
            o.quality_grade,
            
            COUNT(DISTINCT e.id) as event_count,
            COUNT(DISTINCT ir.id) as issue_count,
            SUM(ru.cost) as total_resource_cost
            
        FROM producers p
        JOIN production_units u ON u.producer_id = p.id
        JOIN activities a ON a.production_unit_id = u.id
        LEFT JOIN production_outcomes o ON o.activity_id = a.id
        LEFT JOIN activity_events e ON e.activity_id = a.id
        LEFT JOIN issue_reports ir ON ir.activity_id = a.id
        LEFT JOIN resource_usage ru ON ru.activity_id = a.id
        
        WHERE a.status = 'completed'
        AND o.id IS NOT NULL
        
        GROUP BY p.id, u.id, a.id, o.id
    `);
    
    return data;
}
```

This ML/EDA-ready schema enables:
- **Predictive modeling**: Crop yield prediction, pest outbreak forecasting, market price prediction
- **Causal analysis**: Impact of interventions, weather effects, resource optimization
- **Segmentation**: Farmer clustering, activity profiling, risk stratification
- **Recommendation systems**: Personalized crop suggestions, optimal resource allocation
- **Anomaly detection**: Early warning for crop stress, unusual mortality patterns
- **Time-series forecasting**: Seasonal planning, market timing, resource demand



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



## AWS Implementation Guide

### AWS Service Selection Rationale

**Why AWS for Crop Advisory Agent:**
1. **Serverless Architecture**: Pay only for what you use, ideal for variable agricultural workloads
2. **AI/ML Services**: Pre-built services (Rekognition, Bedrock, SageMaker) accelerate development
3. **Global Infrastructure**: Low-latency access across India with Mumbai and Hyderabad regions
4. **Scalability**: Auto-scaling from 10 to 10,000+ users without infrastructure changes
5. **Cost-Effective**: Free tier covers MVP development, pay-as-you-go for production
6. **Managed Services**: Reduce operational overhead, focus on features not infrastructure

### AWS Architecture Patterns

**Serverless Event-Driven Architecture**:
```
User Action → API Gateway → Lambda → DynamoDB
                                ↓
                          EventBridge → Lambda (Async Processing)
                                ↓
                          SNS/SQS → Notifications
```

**Benefits**:
- No server management
- Automatic scaling
- Pay per request
- High availability built-in
- Fast development cycle

### AWS Service Deep Dive

**1. AWS Lambda - Compute Layer**
- **Use Case**: All business logic (crop recommendation, irrigation planning, risk assessment)
- **Configuration**: Python 3.11, 512MB-1GB memory, 30-60s timeout
- **Cost**: ~$0.20 per 1M requests (after free tier)
- **Optimization**: Use Lambda Layers for shared dependencies, enable X-Ray for tracing

**2. Amazon API Gateway - API Management**
- **Use Case**: RESTful API endpoints, WebSocket for real-time updates
- **Configuration**: REST API with Cognito authorizer, rate limiting (1000 req/sec)
- **Cost**: ~$3.50 per 1M requests
- **Optimization**: Enable caching (5-minute TTL), use CloudFront for edge caching

**3. Amazon DynamoDB - NoSQL Database**
- **Use Case**: User profiles, activities, events, sessions
- **Configuration**: On-demand billing, GSI for queries, TTL for temporary data
- **Cost**: ~$1.25 per million writes, $0.25 per million reads
- **Optimization**: Use single-table design, batch operations, DynamoDB Streams for triggers

**4. Amazon S3 - Object Storage**
- **Use Case**: Images, audio files, ML models, backups
- **Configuration**: Standard storage with Intelligent-Tiering
- **Cost**: ~$0.023 per GB/month
- **Optimization**: Use S3 Transfer Acceleration, lifecycle policies for old data

**5. Amazon Rekognition - Image Analysis**
- **Use Case**: Pest detection from crop images
- **Configuration**: Custom Labels project with 10-15 pest classes
- **Cost**: ~$1 per 1000 images analyzed
- **Optimization**: Resize images before analysis, cache results in DynamoDB

**6. Amazon Bedrock - Generative AI**
- **Use Case**: Conversational AI for natural language queries
- **Configuration**: Claude v2 model, 300 token limit per response
- **Cost**: ~$0.01 per 1000 tokens
- **Optimization**: Use prompt caching, limit response length

**7. Amazon SageMaker - ML Platform**
- **Use Case**: Custom crop recommendation models, advanced pest detection
- **Configuration**: ml.t2.medium for inference, ml.p3.2xlarge for training
- **Cost**: ~$0.05/hour (inference), ~$3/hour (training)
- **Optimization**: Use Serverless Inference for variable traffic

**8. Amazon ElastiCache - Caching Layer**
- **Use Case**: Weather data, market prices, frequent queries
- **Configuration**: Redis, cache.t3.micro (0.5 GB memory)
- **Cost**: ~$15/month
- **Optimization**: Set appropriate TTLs, use Redis Cluster for high availability

**9. Amazon SNS - Notifications**
- **Use Case**: Push notifications, SMS alerts
- **Configuration**: Standard topics, mobile push endpoints
- **Cost**: ~$0.50 per 1M publishes, $0.75 per 100 SMS (India)
- **Optimization**: Batch notifications, use topic filters

**10. Amazon Cognito - Authentication**
- **Use Case**: User registration, login, token management
- **Configuration**: User pool with phone number verification
- **Cost**: Free for first 50,000 MAUs
- **Optimization**: Use refresh tokens, implement token caching

### AWS Deployment Strategy

**Infrastructure as Code (AWS SAM)**:
```bash
# Install AWS SAM CLI
pip install aws-sam-cli

# Initialize project
sam init --runtime python3.11 --name crop-advisory-agent

# Build
sam build

# Deploy
sam deploy --guided
```

**CI/CD Pipeline (AWS CodePipeline)**:
```yaml
# buildspec.yml
version: 0.2
phases:
  install:
    runtime-versions:
      python: 3.11
  pre_build:
    commands:
      - pip install -r requirements.txt
      - python -m pytest tests/
  build:
    commands:
      - sam build
  post_build:
    commands:
      - sam deploy --no-confirm-changeset
```

### AWS Cost Estimation

**MVP Phase (Month 1-3):**
- Lambda: Free tier (1M requests)
- DynamoDB: Free tier (25 GB)
- S3: Free tier (5 GB)
- API Gateway: Free tier (1M calls)
- Rekognition: ~$10-20
- ElastiCache: ~$15
- SNS: ~$5
- **Total: ~$30-40/month**

**Growth Phase (100-1000 users):**
- Lambda: ~$10
- DynamoDB: ~$20
- S3: ~$5
- API Gateway: ~$10
- Rekognition: ~$50
- ElastiCache: ~$30
- SNS: ~$20
- SageMaker: ~$50
- **Total: ~$195/month**

**Scale Phase (10,000+ users):**
- Lambda: ~$100
- DynamoDB: ~$200
- S3: ~$50
- API Gateway: ~$100
- Rekognition: ~$500
- ElastiCache: ~$100 (cluster mode)
- SNS: ~$200
- SageMaker: ~$500
- CloudFront: ~$100
- **Total: ~$1,850/month**

### AWS Security Best Practices

**1. IAM Least Privilege**:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:Query"
    ],
    "Resource": "arn:aws:dynamodb:region:account:table/UserProfiles"
  }]
}
```

**2. Secrets Management**:
- Store API keys in AWS Secrets Manager
- Rotate secrets automatically
- Use IAM roles for service-to-service auth

**3. Data Encryption**:
- Enable encryption at rest (DynamoDB, S3)
- Use HTTPS/TLS for data in transit
- Encrypt sensitive fields in application layer

**4. Network Security**:
- Use VPC for ElastiCache
- Security groups for resource isolation
- AWS WAF for API Gateway protection

**5. Monitoring & Compliance**:
- Enable CloudTrail for audit logs
- Use AWS Config for compliance checking
- Set up CloudWatch alarms for anomalies

### AWS Performance Optimization

**1. Lambda Optimization**:
- Use Lambda Layers for dependencies
- Enable Provisioned Concurrency for critical functions
- Optimize cold starts with smaller deployment packages

**2. DynamoDB Optimization**:
- Use single-table design pattern
- Implement efficient GSI queries
- Enable DynamoDB Accelerator (DAX) for read-heavy workloads

**3. API Gateway Optimization**:
- Enable response caching (5-minute TTL)
- Use CloudFront for edge caching
- Implement request throttling

**4. S3 Optimization**:
- Use CloudFront for image delivery
- Enable Transfer Acceleration for uploads
- Implement multipart uploads for large files

**5. Caching Strategy**:
- ElastiCache for weather data (6-hour TTL)
- API Gateway cache for recommendations (5-minute TTL)
- CloudFront cache for static assets (24-hour TTL)

### AWS Monitoring & Observability

**CloudWatch Dashboards**:
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Create custom dashboard
cloudwatch.put_dashboard(
    DashboardName='CropAdvisoryMetrics',
    DashboardBody=json.dumps({
        'widgets': [
            {
                'type': 'metric',
                'properties': {
                    'metrics': [
                        ['AWS/Lambda', 'Invocations', {'stat': 'Sum'}],
                        ['AWS/Lambda', 'Errors', {'stat': 'Sum'}],
                        ['AWS/Lambda', 'Duration', {'stat': 'Average'}]
                    ],
                    'period': 300,
                    'stat': 'Average',
                    'region': 'ap-south-1',
                    'title': 'Lambda Metrics'
                }
            }
        ]
    })
)
```

**CloudWatch Alarms**:
```python
# Create alarm for Lambda errors
cloudwatch.put_metric_alarm(
    AlarmName='HighLambdaErrors',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=1,
    MetricName='Errors',
    Namespace='AWS/Lambda',
    Period=300,
    Statistic='Sum',
    Threshold=10,
    ActionsEnabled=True,
    AlarmActions=['arn:aws:sns:region:account:alerts']
)
```

**X-Ray Tracing**:
```python
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

# Patch all supported libraries
patch_all()

@xray_recorder.capture('crop_recommendation')
def lambda_handler(event, context):
    # Your code here
    pass
```

### AWS Disaster Recovery

**Backup Strategy**:
- DynamoDB: Enable Point-in-Time Recovery (PITR)
- S3: Enable versioning and cross-region replication
- RDS: Automated daily backups with 7-day retention

**High Availability**:
- Multi-AZ deployment for ElastiCache
- DynamoDB global tables for multi-region
- CloudFront for edge caching and failover

**Recovery Procedures**:
1. DynamoDB restore from PITR (RPO: 5 minutes, RTO: 30 minutes)
2. S3 restore from versioning (RPO: 0, RTO: 5 minutes)
3. Lambda automatic retry with exponential backoff

### AWS Development Workflow

**Local Development**:
```bash
# Use SAM Local for testing
sam local start-api

# Invoke function locally
sam local invoke CropRecommenderFunction --event events/test-event.json

# Generate sample events
sam local generate-event apigateway aws-proxy > events/api-event.json
```

**Testing Strategy**:
```python
# Unit tests with moto (AWS mocking)
import boto3
from moto import mock_dynamodb, mock_s3

@mock_dynamodb
def test_save_user_profile():
    # Create mock DynamoDB table
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.create_table(
        TableName='UserProfiles',
        KeySchema=[{'AttributeName': 'user_id', 'KeyType': 'HASH'}],
        AttributeDefinitions=[{'AttributeName': 'user_id', 'AttributeType': 'S'}],
        BillingMode='PAY_PER_REQUEST'
    )
    
    # Test your function
    result = save_user_profile({'user_id': '123', 'name': 'Test'})
    assert result['statusCode'] == 200
```

**Deployment Stages**:
1. **Dev**: Automatic deployment on commit to dev branch
2. **Staging**: Manual approval after dev testing
3. **Production**: Manual approval with rollback capability

This AWS-based architecture provides a scalable, cost-effective, and maintainable solution for the Crop Advisory Agent, leveraging managed services to minimize operational overhead while maximizing development velocity.
