---
author: Satyan Sharma
title: JavaScript Quirk: Why Your Object Keys Are Colliding and Becoming "[object Object]"
date: 2025-07-30
math: true
tags: ["WebDev"]
---

Have you ever tried to use an object as a key in another JavaScript object, only to find that your values are mysteriously overwritten? You're not alone. This is a common stumbling block for developers, and it all revolves around a seemingly magical string: `"[object Object]"`.

Let's dive into what's happening and how to fix it.

## The Problem: When Objects Aren't Unique Keys

Imagine this scenario:

```javascript
const cache = {};
const user1 = { id: 1, name: 'Alice' };
const user2 = { id: 2, name: 'Bob' };

// Try to store data keyed by user objects
cache[user1] = 'Data for Alice';
cache[user2] = 'Data for Bob';

console.log(cache[user1]); // What will this output?
```

You might expect `'Data for Alice'`, but you'll actually get `'Data for Bob'`. Why?

## The Culprit: Implicit toString() Conversion

In JavaScript, object keys are always **strings** (or Symbols). When you try to use an object as a key, JavaScript needs to convert it to a string. It does this by calling the object's `.toString()` method.

For regular objects, the default `.toString()` method returns `"[object Object]"`.

So what's really happening:

```javascript
cache[user1] = 'Data for Alice';
// becomes: cache["[object Object]"] = 'Data for Alice'

cache[user2] = 'Data for Bob';  
// becomes: cache["[object Object]"] = 'Data for Bob'
// This overwrites the previous value!

console.log(cache[user1]);
// becomes: console.log(cache["[object Object]"])
```

Both objects become the same string key, causing a collision!

## Understanding "[object Object]"

This string isn't random—it follows a specific pattern:
- `object` is a fixed label
- `Object` indicates the type of object

Different object types produce different strings:
```javascript
console.log([].toString());        // "[object Array]"
console.log(new Date().toString()); // "[object Date]" 
console.log(function(){}.toString()); // "[object Function]"
```

But all plain objects (`{}` or `new Object()`) return `"[object Object]"`.

## The Solution: Use the Right Tool for the Job

### 1. Map - The Modern Solution

ES6 introduced `Map`, which allows **any value** as a key, including objects:

```javascript
const cache = new Map();
const user1 = { id: 1, name: 'Alice' };
const user2 = { id: 2, name: 'Bob' };

cache.set(user1, 'Data for Alice');
cache.set(user2, 'Data for Bob');

console.log(cache.get(user1)); // 'Data for Alice' ✅
console.log(cache.get(user2)); // 'Data for Bob' ✅
```

### 2. Unique Identifiers

If you must use regular objects, use a unique property as the key:

```javascript
const cache = {};
const user1 = { id: 1, name: 'Alice' };
const user2 = { id: 2, name: 'Bob' };

cache[user1.id] = 'Data for Alice';
cache[user2.id] = 'Data for Bob';

console.log(cache[1]); // 'Data for Alice' ✅
console.log(cache[2]); // 'Data for Bob' ✅
```

### 3. JSON Stringification (With Caution)

For complex objects, you could use JSON.stringify(), but beware of key order and non-JSON-safe values:

```javascript
const cache = {};
const user1 = { id: 1, name: 'Alice' };
const user2 = { id: 2, name: 'Bob' };

cache[JSON.stringify(user1)] = 'Data for Alice';
cache[JSON.stringify(user2)] = 'Data for Bob';

console.log(cache[JSON.stringify(user1)]); // 'Data for Alice' ✅
```

## When You Might Actually Want "[object Object]"

This behavior isn't always a bug—sometimes it's a feature! If you want to store metadata about object types rather than specific instances, this pattern can be useful:

```javascript
const typeHandlers = {
  "[object Array]": (arr) => `Processing array with ${arr.length} items`,
  "[object Date]": (date) => `Processing date: ${date.toISOString()}`,
  "[object Object]": (obj) => `Processing plain object`
};

function processValue(value) {
  const typeString = Object.prototype.toString.call(value);
  return typeHandlers[typeString]?.(value) || 'Unknown type';
}
```

## Key Takeaways

1. **Object keys are always strings** (or Symbols) in JavaScript
2. **Objects used as keys get converted** to `"[object Object]"` via `.toString()`
3. **All plain objects become the same key**, causing collisions
4. **Use `Map`** when you need object keys
5. **Use unique identifiers** when working with regular objects

Understanding this quirk will save you from hours of debugging mysterious value overwrites. The next time you see `"[object Object]"`, you'll know exactly what's happening and how to handle it!

---
