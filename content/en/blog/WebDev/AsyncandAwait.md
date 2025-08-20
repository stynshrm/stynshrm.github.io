---
author: Satyan Sharma
title: Async & Await in JavaScript and TypeScript
date: 2025-07-29
math: true
tags: ["WebDev"]
---


The `async` and `await` are fundamental concepts in modern JavaScript/TypeScript. Let me explain when and why to use them.

## �� **What Are `async` and `await`?**

### **`async`** = "This function will do something that takes time"
### **`await`** = "Wait for this slow operation to finish before continuing"

## �� **When to Use Them**

### **1. Database Operations (Always Async)**
```typescript
// ❌ Wrong - Database operations are always async
const user = User.findById(id); // This returns a Promise, not the user!

// ✅ Correct - Use await to get the actual result
const user = await User.findById(id); // Now you get the actual user
```

### **2. File Operations**
```typescript
// ❌ Wrong
const data = fs.readFile('file.txt'); // Returns Promise

// ✅ Correct
const data = await fs.readFile('file.txt'); // Gets actual file content
```

### **3. API Calls**
```typescript
// ❌ Wrong
const response = fetch('https://api.example.com/data'); // Returns Promise

// ✅ Correct
const response = await fetch('https://api.example.com/data'); // Gets actual response
```

### **4. Password Hashing**
```typescript
// ❌ Wrong
const hashedPassword = bcrypt.hash(password, 12); // Returns Promise

// ✅ Correct
const hashedPassword = await bcrypt.hash(password, 12); // Gets actual hash
```

## 🎯 **Real Examples from Our Code**

### **In Our User Model:**
```typescript
// ✅ Pre-save middleware - async because bcrypt is slow
userSchema.pre('save', async function(next) {
  if (!this.isModified('password')) {
    return next();
  }
  
  try {
    // bcrypt operations are async
    const salt = await bcrypt.genSalt(12);
    this.password = await bcrypt.hash(this.password, salt);
    next();
  } catch (error) {
    next(error as Error);
  }
});

// ✅ Password comparison - async because bcrypt is slow
userSchema.methods.comparePassword = async function(candidatePassword: string): Promise<boolean> {
  try {
    return await bcrypt.compare(candidatePassword, this.password);
  } catch (error) {
    return false;
  }
};
```

### **In Our Test Function:**
```typescript
// ✅ Test function - async because database operations are slow
export const testUserModel = async () => {
  try {
    // Database operations are async
    const testUser = new User({...});
    await testUser.save(); // Wait for save to complete
    
    const savedUser = await User.findById(testUser._id); // Wait for find to complete
    await User.findByIdAndDelete(testUser._id); // Wait for delete to complete
    
    return true;
  } catch (error) {
    return false;
  }
};
```

## 🎯 **When NOT to Use `async`/`await`**

### **1. Simple Calculations (Synchronous)**
```typescript
// ✅ No async needed - simple math
function add(a: number, b: number): number {
  return a + b;
}

// ✅ No async needed - string manipulation
function getFullName(firstName: string, lastName: string): string {
  return `${firstName} ${lastName}`;
}
```

### **2. Virtual Fields (Synchronous)**
```typescript
// ✅ No async needed - just combining strings
userSchema.virtual('fullName').get(function() {
  return `${this.firstName} ${this.lastName}`;
});
```

### **3. Token Generation (Synchronous)**
```typescript
// ✅ No async needed - just creating random strings
userSchema.methods.generateEmailVerificationToken = function(): string {
  const verificationToken = crypto.randomBytes(32).toString('hex');
  this.emailVerificationToken = verificationToken;
  this.emailVerificationExpires = new Date(Date.now() + 24 * 60 * 60 * 1000);
  return verificationToken;
};
```

## 🎯 **The Rule of Thumb**

### **Use `async`/`await` when:**
- ✅ **Database operations** (find, save, update, delete)
- ✅ **File operations** (read, write)
- ✅ **API calls** (fetch, axios)
- ✅ **Password hashing** (bcrypt)
- ✅ **Email sending** (nodemailer)
- ✅ **Any operation that takes time**

### **Don't use `async`/`await` when:**
- ❌ **Simple calculations** (math, string manipulation)
- ❌ **Creating objects** (new User())
- ❌ **Setting properties** (user.firstName = "John")
- ❌ **Returning values** (return true)

## 🎯 **Common Patterns**

### **1. Try-Catch with Async**
```typescript
async function someFunction() {
  try {
    const result = await someAsyncOperation();
    return result;
  } catch (error) {
    console.error('Error:', error);
    throw error;
  }
}
```

### **2. Multiple Async Operations**
```typescript
// ❌ Wrong - Sequential (slow)
const user1 = await User.findById(id1);
const user2 = await User.findById(id2);
const user3 = await User.findById(id3);

// ✅ Correct - Parallel (fast)
const [user1, user2, user3] = await Promise.all([
  User.findById(id1),
  User.findById(id2),
  User.findById(id3)
]);
```

### **3. Conditional Async**
```typescript
async function processUser(userId: string) {
  const user = await User.findById(userId);
  
  if (user.isEmailVerified) {
    // Only do async operation if needed
    await sendWelcomeEmail(user.email);
  }
  
  return user;
}
```

## 🎯 **In Simple Terms**

- **`async`** = "This function does slow things"
- **`await`** = "Wait for this slow thing to finish"
- **Use them for**: Database, files, APIs, hashing, emails
- **Don't use them for**: Math, strings, simple operations

Think of it like:
- **Without await**: "Go get me a coffee" (you get a promise to get coffee)
- **With await**: "Wait here while I get you a coffee" (you get the actual coffee)

