#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <iomanip>
#include <ctime>
#include <algorithm>
#include <sstream>

using namespace std;

// Forward declarations
class Student;
class Meal;
class Order;
class MealDatabase;

// Meal class representing individual meal items
class Meal {
private:
    string name;
    string category; // breakfast, lunch, supper
    double price;
    int preparationTime; // in minutes
    
public:
    Meal(string n, string cat, double p, int prepTime) 
        : name(n), category(cat), price(p), preparationTime(prepTime) {}
    
    // Getters
    string getName() const { return name; }
    string getCategory() const { return category; }
    double getPrice() const { return price; }
    int getPreparationTime() const { return preparationTime; }
    
    // Display meal information
    void displayMeal() const {
        cout << "- " << name << " (Ksh. " << fixed << setprecision(2) << price 
             << ") - Prep time: " << preparationTime << " mins" << endl;
    }
};

// Student class with authentication and budget management
class Student {
private:
    string name;
    string adminNumber;
    double dailyBudget;
    double spentToday;
    vector<Order> todaysOrders;
    
public:
    Student(string n, string admin) : name(n), adminNumber(admin), dailyBudget(1000.0), spentToday(0.0) {}
    
    // Getters
    string getName() const { return name; }
    string getAdminNumber() const { return adminNumber; }
    double getDailyBudget() const { return dailyBudget; }
    double getSpentToday() const { return spentToday; }
    double getRemainingBudget() const { return dailyBudget - spentToday; }
    
    // Budget management
    bool canAfford(double amount) const {
        return (spentToday + amount) <= dailyBudget;
    }
    
    void addExpense(double amount) {
        spentToday += amount;
    }
    
    void resetDailySpending() {
        spentToday = 0.0;
        todaysOrders.clear();
    }
    
    // Display student info
    void displayBudgetInfo() const {
        cout << "\n=== Budget Information for " << name << " ===" << endl;
        cout << "Daily Budget: Ksh. " << fixed << setprecision(2) << dailyBudget << endl;
        cout << "Spent Today: Ksh. " << fixed << setprecision(2) << spentToday << endl;
        cout << "Remaining: Ksh. " << fixed << setprecision(2) << getRemainingBudget() << endl;
    }
};

// Order class to track meal orders
class Order {
private:
    static int orderCounter;
    int orderNumber;
    string studentName;
    Meal meal;
    string orderTime;
    int waitingTime;
    
public:
    Order(string student, Meal m, int waitTime) 
        : studentName(student), meal(m), waitingTime(waitTime) {
        orderNumber = ++orderCounter;
        
        // Get current time
        time_t now = time(0);
        char* timeStr = ctime(&now);
        orderTime = string(timeStr);
        orderTime.pop_back(); // Remove newline
    }
    
    // Getters
    int getOrderNumber() const { return orderNumber; }
    string getStudentName() const { return studentName; }
    Meal getMeal() const { return meal; }
    int getWaitingTime() const { return waitingTime; }
    string getOrderTime() const { return orderTime; }
    
    void displayOrder() const {
        cout << "\n=== ORDER CONFIRMATION ===" << endl;
        cout << "Order Number: #" << orderNumber << endl;
        cout << "Student: " << studentName << endl;
        cout << "Meal: " << meal.getName() << " (" << meal.getCategory() << ")" << endl;
        cout << "Price: Ksh. " << fixed << setprecision(2) << meal.getPrice() << endl;
        cout << "Order Time: " << orderTime << endl;
        cout << "Estimated Waiting Time: " << waitingTime << " minutes" << endl;
        cout << "=========================" << endl;
    }
};

// Initialize static member
int Order::orderCounter = 0;

// MealDatabase class to manage all meals and timing
class MealDatabase {
private:
    vector<Meal> breakfast;
    vector<Meal> lunch;
    vector<Meal> supper;
    map<string, int> currentWaitTimes; // Category -> wait time in minutes
    
public:
    MealDatabase() {
        initializeMeals();
        initializeWaitTimes();
    }
    
    void initializeMeals() {
        // Breakfast items
        breakfast.push_back(Meal("Mandazi & Tea", "breakfast", 80.0, 5));
        breakfast.push_back(Meal("Bread & Coffee", "breakfast", 100.0, 3));
        breakfast.push_back(Meal("Chapati & Milk Tea", "breakfast", 120.0, 8));
        breakfast.push_back(Meal("Porridge & Eggs", "breakfast", 150.0, 10));
        breakfast.push_back(Meal("Pancakes & Juice", "breakfast", 180.0, 12));
        
        // Lunch items
        lunch.push_back(Meal("Rice & Beans", "lunch", 200.0, 15));
        lunch.push_back(Meal("Ugali & Sukuma Wiki", "lunch", 180.0, 12));
        lunch.push_back(Meal("Chapati & Beef Stew", "lunch", 300.0, 20));
        lunch.push_back(Meal("Pilau & Chicken", "lunch", 350.0, 25));
        lunch.push_back(Meal("Fish & Rice", "lunch", 280.0, 18));
        lunch.push_back(Meal("Githeri & Vegetables", "lunch", 160.0, 10));
        
        // Supper items
        supper.push_back(Meal("Tea & Bread", "supper", 80.0, 5));
        supper.push_back(Meal("Milk & Biscuits", "supper", 100.0, 3));
        supper.push_back(Meal("Light Soup & Bread", "supper", 120.0, 8));
        supper.push_back(Meal("Porridge & Fruits", "supper", 140.0, 7));
        supper.push_back(Meal("Sandwich & Juice", "supper", 160.0, 10));
    }
    
    void initializeWaitTimes() {
        currentWaitTimes["breakfast"] = 10;
        currentWaitTimes["lunch"] = 20;
        currentWaitTimes["supper"] = 8;
    }
    
    void displayMealCategory(const string& category) {
        cout << "\n=== " << category << " MENU ===" << endl;
        
        vector<Meal>* meals = nullptr;
        if (category == "breakfast") meals = &breakfast;
        else if (category == "lunch") meals = &lunch;
        else if (category == "supper") meals = &supper;
        
        if (meals) {
            for (size_t i = 0; i < meals->size(); i++) {
                cout << (i + 1) << ". ";
                (*meals)[i].displayMeal();
            }
        }
        cout << "Current waiting time for " << category << ": " 
             << currentWaitTimes[category] << " minutes" << endl;
    }
    
    Meal* getMeal(const string& category, int index) {
        vector<Meal>* meals = nullptr;
        if (category == "breakfast") meals = &breakfast;
        else if (category == "lunch") meals = &lunch;
        else if (category == "supper") meals = &supper;
        
        if (meals && index >= 0 && static_cast<size_t>(index) < meals->size()) {
            return &(*meals)[index];
        }
        return nullptr;
    }
    
    int getWaitTime(const string& category) {
        return currentWaitTimes[category];
    }
    
    void updateWaitTime(const string& category) {
        // Simulate dynamic wait times based on orders
        currentWaitTimes[category] += 5; // Add 5 minutes per order
        if (currentWaitTimes[category] > 45) {
            currentWaitTimes[category] = 45; // Max 45 minutes
        }
    }
};

// StudentPortal class - main system controller
class StudentPortal {
private:
    vector<Student> students;
    MealDatabase database;
    vector<Order> allOrders;
    
public:
    StudentPortal() {
        initializeStudents();
    }
    
    void initializeStudents() {
        students.push_back(Student("Ray", "ADM001"));
        students.push_back(Student("Lisa", "ADM002"));
        students.push_back(Student("Sandra", "ADM003"));
        students.push_back(Student("Joyline", "ADM004"));
        students.push_back(Student("Hamid", "ADM005"));
    }
    
    Student* authenticateStudent(const string& adminNumber) {
        for (auto& student : students) {
            if (student.getAdminNumber() == adminNumber) {
                return &student;
            }
        }
        return nullptr;
    }
    
    void displayWelcome() {
        cout << "=================================" << endl;
        cout << "   STUDENT MEAL PORTAL SYSTEM   " << endl;
        cout << "=================================" << endl;
        cout << "Welcome to the Student Meal Registration Portal!" << endl;
        cout << "Daily Budget per Student: Ksh. 1000" << endl;
        cout << "=================================" << endl;
    }
    
    void displayMealCategories() {
        cout << "\n=== MEAL CATEGORIES ===" << endl;
        cout << "1. Breakfast" << endl;
        cout << "2. Lunch" << endl;
        cout << "3. Supper" << endl;
        cout << "4. View Budget Information" << endl;
        cout << "5. Exit" << endl;
    }
    
    bool validateBudget(Student* student, double mealPrice) {
        if (!student->canAfford(mealPrice)) {
            cout << "\n❌ BUDGET EXCEEDED!" << endl;
            cout << "Meal price: Ksh. " << fixed << setprecision(2) << mealPrice << endl;
            cout << "Your remaining budget: Ksh. " << fixed << setprecision(2) 
                 << student->getRemainingBudget() << endl;
            cout << "Please choose a meal within your budget." << endl;
            return false;
        }
        return true;
    }
    
    void processMealOrder(Student* student, const string& category) {
        database.displayMealCategory(category);
        
        cout << "\nEnter meal number (or 0 to go back): ";
        int mealChoice;
        cin >> mealChoice;
        
        if (mealChoice == 0) return;
        
        Meal* selectedMeal = database.getMeal(category, mealChoice - 1);
        if (!selectedMeal) {
            cout << "Invalid meal selection!" << endl;
            return;
        }
        
        // Check budget
        if (!validateBudget(student, selectedMeal->getPrice())) {
            return;
        }
        
        // Create order
        int waitTime = database.getWaitTime(category) + selectedMeal->getPreparationTime();
        Order newOrder(student->getName(), *selectedMeal, waitTime);
        
        // Process payment
        student->addExpense(selectedMeal->getPrice());
        allOrders.push_back(newOrder);
        
        // Update wait times
        database.updateWaitTime(category);
        
        // Display confirmation
        newOrder.displayOrder();
        
        cout << "\n✅ Order placed successfully!" << endl;
        cout << "Updated budget - Remaining: Ksh. " << fixed << setprecision(2) 
             << student->getRemainingBudget() << endl;
    }
    
    void run() {
        displayWelcome();
        
        while (true) {
            cout << "\n=== LOGIN ===" << endl;
            cout << "Enter your Administration Number (or 'quit' to exit): ";
            string adminNumber;
            cin >> adminNumber;
            
            if (adminNumber == "quit") {
                cout << "Thank you for using the Student Meal Portal!" << endl;
                break;
            }
            
            Student* currentStudent = authenticateStudent(adminNumber);
            if (!currentStudent) {
                cout << "❌ Invalid Administration Number! Please try again." << endl;
                continue;
            }
            
            cout << "\n✅ Welcome, " << currentStudent->getName() << "!" << endl;
            
            // Student session
            while (true) {
                displayMealCategories();
                cout << "\nSelect an option: ";
                int choice;
                cin >> choice;
                
                switch (choice) {
                    case 1:
                        processMealOrder(currentStudent, "breakfast");
                        break;
                    case 2:
                        processMealOrder(currentStudent, "lunch");
                        break;
                    case 3:
                        processMealOrder(currentStudent, "supper");
                        break;
                    case 4:
                        currentStudent->displayBudgetInfo();
                        break;
                    case 5:
                        cout << "Logging out " << currentStudent->getName() << "..." << endl;
                        goto next_user;
                    default:
                        cout << "Invalid option! Please try again." << endl;
                }
            }
            next_user:;
        }
    }
};

// Main function
int main() {
    cout << "/*" << endl;
    cout << " * STUDENT MEAL PORTAL SYSTEM" << endl;
    cout << " * Developed in C++" << endl;
    cout << " * " << endl;
    cout << " * RECOMMENDED IDEs FOR THIS PROJECT:" << endl;
    cout << " * 1. Visual Studio Code (Free, Cross-platform)" << endl;
    cout << " *    - Excellent C++ extension support" << endl;
    cout << " *    - Integrated terminal and debugging" << endl;
    cout << " *    - Git integration" << endl;
    cout << " * " << endl;
    cout << " * 2. Code::Blocks (Free, Cross-platform)" << endl;
    cout << " *    - Built specifically for C/C++" << endl;
    cout << " *    - Easy project management" << endl;
    cout << " *    - Good for beginners" << endl;
    cout << " * " << endl;
    cout << " * 3. CLion (JetBrains - Paid/Free for students)" << endl;
    cout << " *    - Professional IDE with advanced features" << endl;
    cout << " *    - Excellent debugging and refactoring tools" << endl;
    cout << " *    - Built-in CMake support" << endl;
    cout << " * " << endl;
    cout << " * 4. Dev-C++ (Free, Windows)" << endl;
    cout << " *    - Simple and lightweight" << endl;
    cout << " *    - Good for learning C++" << endl;
    cout << " *" << endl;
    cout << " * COMPILATION:" << endl;
    cout << " * g++ -o student_portal student_meal_portal.cpp" << endl;
    cout << " * ./student_portal" << endl;
    cout << " */" << endl << endl;
    
    // Run the portal system
    StudentPortal portal;
    portal.run();
    
    return 0;
}