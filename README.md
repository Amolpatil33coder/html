# html




Here’s the **code** from the image you uploaded — clean and ready to copy and use 👇

---

### 🗂 Step 1: Create project folder

```bash
mkdir docker-webapp
cd docker-webapp
```

---

### 📝 Step 2: Create `index.html`

```bash
notepad index.html
```

**index.html**

```html
<!DOCTYPE html>
<html>
<head>
    <title>My First Docker Web App</title>
</head>
<body>
    <h1>Hello, Docker!</h1>
    <p>This is a sample web application running in Docker.</p>
</body>
</html>
```

---

### 📝 Step 3: Create `Dockerfile`

```bash
notepad Dockerfile
```

**Dockerfile**

```dockerfile
FROM nginx
COPY index.html /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

---

### 🧱 Step 4: Build Docker Image

```bash
docker build -t mywebapp .
```

---

### 🚀 Step 5: Run Docker Container

```bash
docker run -d -p 8080:80 mywebapp
```

---

### 🌐 Step 6: Open in Browser

Visit:

```
http://localhost:8080
```

---

Would you like me to explain **how this Dockerfile works** (each line)?
