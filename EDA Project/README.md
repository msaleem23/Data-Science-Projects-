# Instacart EDA Project

## What's This Project About?

Have you ever wondered about the shopping habits of Instacart customers? This project digs into the Instacart dataset to uncover insights into customer behavior. By cleaning and analyzing the data, we'll explore patterns in how and when people shop for groceries online.

## The Data

I've got several files that make up our dataset:
- `instacart_orders.csv`: Details about individual orders placed on Instacart.
- `products.csv`: Information about the products available on Instacart.
- `order_products.csv`: Information about products included in each order.
- `aisles.csv`: Details about the aisles in which products are categorized.
- `departments.csv`: Details about the departments in which products are categorized.

### Data Dictionary

#### `instacart_orders.csv`
- `order_id`: Unique identifier for each order
- `user_id`: Unique customer identifier
- `order_number`: Number of orders placed by customer
- `order_dow`: Day of the week the order was placed
- `order_hour_of_day`: Hour of the day the order was placed
- `days_since_prior_order`: Days since previous order

#### `products.csv`
- `product_id`: Unique product identifier
- `product_name`: Product name
- `aisle_id`: Grocery aisle category identifier
- `department_id`: Grocery department category identifier

#### `order_products.csv`
- `order_id`: Unique order identifier
- `product_id`: Unique product identifier
- `add_to_cart_order`: Sequential order of items in cart
- `reordered`: 0 for first-time order, 1 for reorder

#### `aisles.csv`
- `aisle_id`: Grocery aisle category identifier
- `aisle`: Aisle name

#### `departments.csv`
- `department_id`: Grocery department category identifier
- `department`: Department name

## How to Get Through This Project

You'll find a Jupyter Notebook template named `EDA_final_project_template.ipynb` that will guide you through the process. Here's the plan:

### Part 1: Check Out the Data
Open up the data files and see what's inside. Make some notes about what you find right there in the notebook.

### Part 2: Clean Up the Data
Sometimes data can be a bit messy. We'll tidy up things like column names, and handle missing values and duplicates. There'll be spots in the notebook to jot down what you did.

### Part 3: The Real Deal Analysis
This is where it gets juicy. We'll run some code to analyze customer shopping habits, such as:
- Order placement by hour and day
- Time between orders
- Most frequently ordered products
- And more

In the "Conclusion" section of the notebook, we'll discuss what these insights mean.

Think of this project like a detective case. You've got a hunch, and now you're looking for clues to see if you're right. It's the same thing data pros do all the time in their jobs. Ready to play detective? Let's dive in!
