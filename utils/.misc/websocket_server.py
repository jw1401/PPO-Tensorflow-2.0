#!/usr/bin/env python

import asyncio
import websockets

async def echo(websocket, path):
    # name = await websocket.recv()
    # print(name)

    # greeting = "Hello " + name

    await websocket.send("run")
    
    # async for message in websocket:
    #     # name = await websocket.recv()
    #     # print(name)
    #     print(message)
    #     await websocket.send(message)

server = websockets.serve(echo, 'localhost', 8765)
asyncio.get_event_loop().run_until_complete(server)
asyncio.get_event_loop().run_forever()
