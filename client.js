let field = [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0]
]
let param = {
    method: 'POST', 
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name:"hello"})
}
let res = await fetch("http://localhost:8000/", param)
await res.json()