
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>main page</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    <style>
        body {
            background-color: #f8f9fa;
        }
        .navbar {
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }

        .nav-link:hover {
            color: orange !important; /* Change text color on hover */
        }
        .nav-link.btn {
            border-radius: 5px; /* Add rounded corners to buttons */
        }
        /* Additional styles for modal */
        .modal-body {
            text-align: center;
        }
        .modal-content {
            background-color: #fff; /* Set modal content background color */
            color: #000; /* Set modal content text color */
        }
        .modal-content.dark {
            background-color: #000; /* Set modal content background color for dark theme */
            color: #fff; /* Set modal content text color for dark theme */
        }
    </style>
</head>
<body>


<!-- Navbar -->
<nav class="navbar navbar-expand-lg navbar-light bg-light">
  <a class="navbar-brand mx-auto" href="#">Ecommerce Recommendation System</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse justify-content-end" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link" href="http://127.0.0.1:5000"><i class="fas fa-home"></i> Back </a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#" id="settingsLink"><i class="fas fa-cog"></i> Settings</a>
      </li>
      <li class="nav-item">
        <a class="nav-link btn btn-outline-primary mr-2" href="#" data-toggle="modal" data-target="#signupModal">Sign Up</a>
      </li>
      <li class="nav-item">
        <a class="nav-link btn btn-primary" href="#" data-toggle="modal" data-target="#signinModal">Sign In</a>
      </li>
    </ul>
  </div>
</nav>


<!-- Search Bar -->
<div class="container" style="margin-top:30px;">
    <form action="/recommendations" method="post" style="display:flex;">
            <input type="text" class="form-control mr-2" name="prod" placeholder="Search for products...">
            <input type="number" class="form-control" name="nbr" placeholder="Number of products..." style="width:100px;">
            <button class="btn btn-primary">Search</button>
    </form>
</div>

{% if message %}
<h5 style="margin-left:42%;margin-top:10px; text:bold;">{{message}}</h5>
{% endif %}

<!--trending recommendations-->
<!--'category', 'ratings', 'brand', 'image'-->
<!-- Products -->
{% if not content_based_rec.empty %}
<div class="container mt-5">
  <h2 class="text-center mb-4">Recommended Products</h2>
  <div class="row mt-4">
    {% for index, product in content_based_rec.iterrows() %}
    <div class="col-lg-3 col-md-4 col-sm-6 mb-4">
      <div class="card">
  <img src="{{ product['image'] }}" class="card-img-top" alt="{{ product['category'] }}" style="height: 200px;">
  
        <div class="card-body">
          <h5 class="card-title">{{ truncate(product['category'],12) }}</h5>
          <p class="card-text">Brand: {{ product['brand'] }}</p>
          <button type="button" class="btn btn-primary" data-toggle="modal" data-target="#productModal{{ index }}">Buy Now</button>
        </div>
      </div>
    </div>
    <!-- Modal -->
    <div class="modal fade" id="productModal{{ index }}" tabindex="-1" role="dialog" aria-labelledby="productModalLabel" aria-hidden="true">
      <div class="modal-dialog" role="document">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="productModalLabel">{{ product['category'] }}</h5>
            <button type="button" class="close" data-dismiss="modal" aria-label="Close">
              <span aria-hidden="true">&times;</span>
            </button>
          </div>
          <div class="modal-body">
            <div class="row">
              <div class="col-md-4">
                  <img src="{{ product['image'] }}" class="img-fluid" alt="{{ product['category'] }}" style="height: 200px;">
              </div>
              <div class="col-md-8">
                <p><strong>Brand:</strong> {{ product['brand'] }}</p>
                <!-- Add other product details here -->
                <!-- Example: -->
                <p><strong>Ratings:</strong> {{ product['ratings'].values[0] if product['ratings'] is not none else 'N/A' }}</p>
                <p><strong>Price:</strong> {{ (100 + loop.index * 10) }}</p>
                <!-- Add more details as needed -->
              </div>
            </div>
          </div>
          <div class="modal-footer">
            <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
            <button type="button" class="btn btn-primary add-to-cart" data-toggle="modal" data-target="#cartModal" 
            data-name="{{ product['category'] }}" 
            data-names="{{ product['brand'] }}" 
            data-namess="{{ product['image'] }}" 

            data-price="{{ (100 + loop.index * 10) }}">
        Add to Cart
    </button>
    
              </div>
        </div>
      </div>
    </div>
    {% endfor %}
  </div>
</div>

{% endif %}

<!--trending recommendations-->




<!-- Cart Modal -->
<div class="modal fade" id="cartModal" tabindex="-1" role="dialog" aria-labelledby="cartModalLabel" aria-hidden="true">
  <div class="modal-dialog" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="cartModalLabel">Enter Your Details</h5>
        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <form action="http://localhost/osms/addcart.php" method="POST">
        <div class="modal-body">
          <input type="hidden" name="product_name" id="product_name">
          <input type="hidden" name="product_price" id="product_price">
          <input type="hidden" name="product_brand" id="product_brand">
          <input type="hidden" name="product_image" id="product_image">

          <div class="form-group">
            <label for="user_name">Name:</label>
            <input type="text" class="form-control" name="user_name" required>
          </div>
          <div class="form-group">
            <label for="user_email">Email:</label>
            <input type="email" class="form-control" name="user_email" required>
          </div>
          
          <div class="form-group">
            <label for="quantity">Quantity:</label>
            <input type="number" class="form-control" name="quantity" min="1" value="1" required>
          </div>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
          <button type="submit" class="btn btn-primary">Add to Cart</button>
        </div>
      </form>
    </div>
  </div>
</div>




<!-- Settings Modal -->
<div class="modal fade" id="settingsModal" tabindex="-1" role="dialog" aria-labelledby="settingsModalLabel" aria-hidden="true">
  <div class="modal-dialog" role="document">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title" id="settingsModalLabel">Settings</h5>
        <button type="button" class="close" data-dismiss="modal" aria-label="Close">
          <span aria-hidden="true">&times;</span>
        </button>
      </div>
      <div class="modal-body">
        <h5>Choose Theme:</h5>
        <div class="form-check">
          <input class="form-check-input" type="radio" name="theme" id="defaultTheme" value="default" checked>
          <label class="form-check-label" for="defaultTheme">
            Default
          </label>
        </div>
        <div class="form-check">
          <input class="form-check-input" type="radio" name="theme" id="blackTheme" value="black">
          <label class="form-check-label" for="blackTheme">
            Black Theme
          </label>
        </div>
        <div class="form-check">
          <input class="form-check-input" type="radio" name="theme" id="greenTheme" value="green">
          <label class="form-check-label" for="greenTheme">
            Green Theme
          </label>
        </div>
        <hr>
        <h5>Zoom:</h5>
        <button type="button" class="btn btn-primary mr-2" id="zoomIn">Zoom In</button>
        <button type="button" class="btn btn-primary" id="zoomOut">Zoom Out</button>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
        <button type="button" class="btn btn-primary" id="applyTheme">Apply</button>
      </div>
    </div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.5.4/dist/umd/popper.min.js"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
<script>
  // Handle click on Settings link to open the modal
  document.getElementById('settingsLink').addEventListener('click', function() {
    $('#settingsModal').modal('show');
  });

  // Handle theme apply button click
  document.getElementById('applyTheme').addEventListener('click', function() {
    // Get the selected theme value
    var selectedTheme = document.querySelector('input[name="theme"]:checked').value;

    // Apply the selected theme
    if (selectedTheme === 'black') {
      document.body.style.backgroundColor = 'black';
      document.body.style.color = 'white';
    } else if (selectedTheme === 'green') {
      document.body.style.backgroundColor = 'green';
      document.body.style.color = 'white';
    } else {
      // Default theme
      document.body.style.backgroundColor = '#f8f9fa';
      document.body.style.color = 'black';
    }

    // Close the modal
    $('#settingsModal').modal('hide');
  });

  // Handle zoom in button click
  document.getElementById('zoomIn').addEventListener('click', function() {
    document.body.style.zoom = "115%";
  });

  // Handle zoom out button click
  document.getElementById('zoomOut').addEventListener('click', function() {
    document.body.style.zoom = "100%";
  });
</script>


<!--footer-->
<footer class="footer bg-dark text-white" style="margin-top:300px;">
    <div class="container" style="padding-top:20px;">
        <div class="row">
            <div class="col-md-3 col-sm-6">
                <h5>About Us</h5>
                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla pretium risus quis urna maximus, eget vestibulum risus vestibulum.</p>
            </div>
            <div class="col-md-3 col-sm-6">
                <h5>Quick Links</h5>
                <ul class="list-unstyled">
                    <li><a href="#">Home</a></li>
                    <li><a href="#">About</a></li>
                    <li><a href="#">Services</a></li>
                    <li><a href="#">Contact</a></li>
                </ul>
            </div>
            <div class="col-md-3 col-sm-6">
                <h5>Support</h5>
                <ul class="list-unstyled">
                    <li><a href="#">FAQ</a></li>
                    <li><a href="#">Terms of Service</a></li>
                    <li><a href="#">Privacy Policy</a></li>
                </ul>
            </div>
            <div class="col-md-3 col-sm-6">
                <h5>Contact Us</h5>
                <address>
                    <strong>Company Name</strong><br>
                    123 Street, City<br>
                    Country<br>
                    <i class="fas fa-phone"></i> Phone: +1234567890<br>
                    <i class="fas fa-envelope"></i> Email: info@example.com
                </address>
            </div>
        </div>
        <div class="row mt-3">
            <div class="col-md-12">
                <hr class="bg-light">
                <p class="text-center">© 2024 Ecommerce Recommendation System. All Rights Reserved.</p>
            </div>
        </div>
    </div>
</footer>
<script>
  // When "Add to Cart" button is clicked
  document.addEventListener("DOMContentLoaded", function () {
      let addToCartButtons = document.querySelectorAll(".add-to-cart");
      addToCartButtons.forEach(button => {
          button.addEventListener("click", function () {
              let productName = this.getAttribute("data-name");
              let productPrice = this.getAttribute("data-price");
              let productbrand = this.getAttribute("data-names");
              let productimage = this.getAttribute("data-namess");

              // Populate the hidden input fields in the cart modal
              document.getElementById("product_name").value = productName;
              document.getElementById("product_price").value = productPrice;
              document.getElementById("product_brand").value = productbrand;
              document.getElementById("product_image").value = productimage;


          });
      });
  });
</script>


</body>
</html>