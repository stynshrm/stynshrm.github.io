---
author: Satyan Sharma
title: Understanding JavaScript Async- Callbacks, Promises, and Async/Await
date: 2025-07-29
math: true
tags: ["WebDev"]
---

# Understanding JavaScript Async: Callbacks, Promises, and Async/Await

JavaScript is single-threaded, but it’s designed to handle asynchronous operations efficiently. In this post, we’ll explore how callbacks, Promises, and `async/await` work, using a simple example of fetching a joke from an API.

---

## 1. Callbacks

A **callback** is a function passed into another function to be executed later.

```javascript
function getJokeWithCallback() {
  fetch("https://official-joke-api.appspot.com/random_joke")
    .then(function handleResponse(response) {
      return response.json();
    })
    .then(function handleData(data) {
      console.log(`${data.setup} - ${data.punchline}`);
    })
    .catch(function handleError(error) {
      console.error("Error:", error);
    });
}

getJokeWithCallback();
```

**Key Points:**

* Each `.then()` and `.catch()` receives a **callback function**.
* The argument of the callback (like `data` or `error`) is provided by the Promise.
* The callbacks are not executed immediately—they’re called **later**, when the Promise resolves or rejects.

---

## 2. Promises

A **Promise** is an object representing a value that may not be available yet but will be resolved in the future.

* `fetch()` returns a Promise.
* `.json()` also returns a Promise because parsing can take time.

**Analogy:** Think of each `.then()` as a student in a classroom passing along the result once it arrives.

---

## 3. Async/Await

`async/await` is syntactic sugar over Promises that makes asynchronous code look synchronous.

```javascript
async function getJokeWithAsync() {
  try {
    const response = await fetch("https://official-joke-api.appspot.com/random_joke");
    const data = await response.json();
    console.log(`${data.setup} - ${data.punchline}`);
  } catch (error) {
    console.error("Error:", error);
  }
}

getJokeWithAsync();
```

**Key Points:**

* `await` pauses the async function until the Promise resolves.
* Both `fetch()` and `response.json()` are asynchronous; they may take time.
* Using `async/await` improves readability compared to nested `.then()` chains.

---

## 4. Named Functions vs Inline Arrow Functions

You can define callbacks outside the main function for clarity:

```javascript
function handleResponse(response) { return response.json(); }
function handleData(data) { console.log(data.setup + " - " + data.punchline); }
function handleError(error) { console.error(error); }

fetch("https://official-joke-api.appspot.com/random_joke")
  .then(handleResponse)
  .then(handleData)
  .catch(handleError);
```

* Arrow functions and named functions are **both valid callbacks**.
* Using named functions can make debugging and code organization easier.

---

## 5. Functions Returning Promises

Some functions **always return a Promise**, either because they are asynchronous or marked `async`.

| Function Type           | Example                                                                          | Returns Promise? |
| ----------------------- | -------------------------------------------------------------------------------- | ---------------- |
| Normal function         | `function add(a, b) { return a + b; }`                                           | No               |
| Async function          | `async function getData() { return 42; }`                                        | Yes              |
| Built-in async function | `fetch("url")`                                                                   | Yes              |
| Promise wrapper         | `function delay(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }` | Yes              |

**Key Rule:** Any `async` function **automatically returns a Promise**, and any value returned inside it is wrapped in that Promise.

---

## 6. Summary

* **Callbacks:** Functions passed into other functions, executed later.
* **Promises:** Objects representing future values; chainable with `.then()` and `.catch()`.
* **Async/Await:** Cleaner syntax for sequential async operations, still uses Promises under the hood.
* **Functions returning Promises:** Include `fetch()`, `.json()`, async functions, and custom Promise wrappers.

Understanding this flow helps avoid “callback hell” and makes it easier to reason about asynchronous JavaScript.


Do you want me to add that diagram too?
