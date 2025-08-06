# Student Meal Portal System

A comprehensive C++ application for managing student meal registrations with budget tracking, authentication, and order management.

## 🎯 Features

### Core Functionality
- **Student Authentication**: Secure login using administration numbers
- **Budget Management**: Daily budget tracking (Ksh. 1000 per student)
- **Meal Categories**: Breakfast, Lunch, and Supper options
- **Real-time Pricing**: Dynamic meal pricing with budget validation
- **Order Management**: Order tracking with confirmation numbers and wait times
- **Time Management**: Preparation and waiting time calculations

### Student Profiles
The system includes 5 pre-registered students:
- **Ray** (ADM001)
- **Lisa** (ADM002) 
- **Sandra** (ADM003)
- **Joyline** (ADM004)
- **Hamid** (ADM005)

### Meal Database
**Breakfast Menu** (5:00 AM - 10:00 AM):
- Mandazi & Tea - Ksh. 80
- Bread & Coffee - Ksh. 100
- Chapati & Milk Tea - Ksh. 120
- Porridge & Eggs - Ksh. 150
- Pancakes & Juice - Ksh. 180

**Lunch Menu** (12:00 PM - 3:00 PM):
- Rice & Beans - Ksh. 200
- Ugali & Sukuma Wiki - Ksh. 180
- Chapati & Beef Stew - Ksh. 300
- Pilau & Chicken - Ksh. 350
- Fish & Rice - Ksh. 280
- Githeri & Vegetables - Ksh. 160

**Supper Menu** (6:00 PM - 9:00 PM):
- Tea & Bread - Ksh. 80
- Milk & Biscuits - Ksh. 100
- Light Soup & Bread - Ksh. 120
- Porridge & Fruits - Ksh. 140
- Sandwich & Juice - Ksh. 160

## 🚀 Getting Started

### Prerequisites
- C++ Compiler (GCC 7.0+ recommended)
- Terminal/Command Line access

### Compilation & Setup

#### Option 1: Quick Start
```bash
# Compile the program
g++ -o student_portal student_meal_portal.cpp

# Run the application
./student_portal
```

#### Option 2: With Optimization
```bash
# Compile with optimizations
g++ -std=c++17 -O2 -Wall -o student_portal student_meal_portal.cpp

# Run the application  
./student_portal
```

### 🔧 Recommended IDEs

#### 1. **Visual Studio Code** (Recommended for Beginners)
- **Cost**: Free
- **Platform**: Cross-platform (Windows, macOS, Linux)
- **Advantages**:
  - Excellent C++ extension support
  - Integrated terminal and debugging
  - Git integration
  - IntelliSense code completion
- **Setup**: Install C++ extension pack

#### 2. **Code::Blocks**
- **Cost**: Free
- **Platform**: Cross-platform
- **Advantages**:
  - Built specifically for C/C++
  - Easy project management
  - Good for beginners
  - No complex configuration needed

#### 3. **CLion** (Best for Advanced Development)
- **Cost**: Paid (Free for students with GitHub Student Pack)
- **Platform**: Cross-platform
- **Advantages**:
  - Professional IDE with advanced features
  - Excellent debugging and refactoring tools
  - Built-in CMake support
  - Advanced code analysis

#### 4. **Dev-C++** (Windows Only)
- **Cost**: Free
- **Platform**: Windows only
- **Advantages**:
  - Simple and lightweight
  - Good for learning C++
  - Portable version available

## 💻 Usage Guide

### 1. Starting the Application
Run the compiled program to see the welcome screen with IDE recommendations and system information.

### 2. User Authentication
```
=== LOGIN ===
Enter your Administration Number (or 'quit' to exit): ADM001
```
Use one of the predefined admin numbers: ADM001, ADM002, ADM003, ADM004, or ADM005.

### 3. Main Menu Navigation
After successful login, you'll see:
```
=== MEAL CATEGORIES ===
1. Breakfast
2. Lunch  
3. Supper
4. View Budget Information
5. Exit
```

### 4. Ordering Process
1. Select meal category (1-3)
2. Browse available meals with prices
3. Enter meal number to order
4. System validates budget automatically
5. Receive order confirmation with:
   - Order number
   - Estimated waiting time
   - Updated budget information

### 5. Budget Tracking
- Each student starts with Ksh. 1000 daily budget
- Real-time budget validation prevents overspending
- View remaining budget anytime via option 4

## 🏗️ System Architecture

### Class Structure

#### `Student` Class
- **Purpose**: Manages student data and budget tracking
- **Key Methods**:
  - `canAfford()`: Budget validation
  - `addExpense()`: Expense tracking
  - `displayBudgetInfo()`: Budget display

#### `Meal` Class  
- **Purpose**: Represents individual meal items
- **Properties**: Name, category, price, preparation time
- **Methods**: Display formatting and getters

#### `Order` Class
- **Purpose**: Tracks meal orders and timing
- **Features**: Unique order numbers, timestamps, wait times
- **Methods**: Order confirmation display

#### `MealDatabase` Class
- **Purpose**: Manages meal inventory and timing
- **Features**: 
  - Categorized meal storage
  - Dynamic wait time calculation
  - Menu display functionality

#### `StudentPortal` Class
- **Purpose**: Main system controller
- **Responsibilities**:
  - User authentication
  - Order processing
  - Budget validation
  - System navigation

## 📊 System Specifications

### Budget Management
- **Daily Budget**: Ksh. 1000 per student
- **Currency**: Kenyan Shillings (Ksh.)
- **Validation**: Real-time budget checking
- **Reset**: Daily automatic reset capability

### Timing System
- **Order Tracking**: Unique sequential order numbers
- **Wait Times**: Dynamic calculation based on:
  - Base category wait time
  - Individual meal preparation time
  - Current kitchen load
- **Time Display**: Estimated serving time in minutes

### Data Storage
- **In-Memory Storage**: All data stored in vectors and maps
- **Persistence**: Session-based (resets on restart)
- **Scalability**: Easy to extend to file/database storage

## 🔧 Customization Options

### Adding New Students
Modify the `initializeStudents()` method in `StudentPortal` class:
```cpp
students.push_back(Student("NewName", "ADM006"));
```

### Adding New Meals
Modify the `initializeMeals()` method in `MealDatabase` class:
```cpp
breakfast.push_back(Meal("New Meal", "breakfast", 150.0, 10));
```

### Changing Budget Limits
Modify the constructor in `Student` class:
```cpp
Student(string n, string admin) : name(n), adminNumber(admin), dailyBudget(1500.0), spentToday(0.0) {}
```

## 🚨 Error Handling

The system includes comprehensive error handling for:
- Invalid administration numbers
- Budget exceeded scenarios
- Invalid meal selections
- Input validation errors

## 🎯 Future Enhancements

Potential improvements for the system:
1. **Database Integration**: MySQL/SQLite for data persistence
2. **Web Interface**: Convert to web-based system
3. **Payment Integration**: M-Pesa or card payment options
4. **Nutritional Information**: Calorie and nutrition tracking
5. **Admin Panel**: Meal management and reporting
6. **Mobile App**: Native mobile application
7. **Queue Management**: Advanced queue and timing system

## 📝 Sample Usage Session

```
=== LOGIN ===
Enter your Administration Number: ADM001
✅ Welcome, Ray!

=== MEAL CATEGORIES ===
Select an option: 2

=== lunch MENU ===
1. Rice & Beans (Ksh. 200.00) - Prep time: 15 mins
2. Ugali & Sukuma Wiki (Ksh. 180.00) - Prep time: 12 mins
...

Enter meal number: 1

=== ORDER CONFIRMATION ===
Order Number: #1
Student: Ray
Meal: Rice & Beans (lunch)
Price: Ksh. 200.00
Estimated Waiting Time: 35 minutes
=========================

✅ Order placed successfully!
Updated budget - Remaining: Ksh. 800.00
```

## 📞 Support

For technical support or questions about the system:
1. Check the error messages for guidance
2. Verify compilation with latest GCC version
3. Ensure all required headers are available
4. Review the code comments for implementation details

---

**Developed in C++** | **Educational Project** | **Version 1.0**