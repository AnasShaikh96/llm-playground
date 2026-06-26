
## Project Name : inspiredIt

This project is more for me than anyone else out there. I have found myself extremely cornered in the sense of taking up projects single handedly because i dont know about designing much, I mean yeah I do know the basics, but what is fun in that? I tried manipulating some designers to join me out but their goal didnt aligned with mine so here I am trying something really out of the box.



## Project Idea and Details

The high level understanding is that a User should be able to create their website by speaking to an AI agent and submitting their reference sites. Once the barebones is ready, he can then demand for standalone features or a section to be edited in a certain way etc.

---
Think of this as a dry flow:

- I'm a GenZ Sales guy.
- I want to create a resume website for me.
- I go to inspiredIt and start the prompt (It looks like any other llm btw)
>Hey I want to build a website for me so that people can see my work and achievements, also I should be able to blog from this website.

Now even in this one line texts there are valuable infos being "Build a website, blog capable, work and achievements".
The AI can funnel it further by asking relevant questions in a bunch such as "Do you want to keep the mood of this website as professional or casual or both?"
or any other funneling question. There are a lot of things that can be taken forward with this.

>Yeah sure, i want to keep this website as little bit both. All in all I'm putting myself out there.

Once this sort of QnA completes, we can then ask for Users references. Now these can be a single website or multiple of them. Depends on the User, but golden rule is allowing atleast 3 websites. 

We can also ask them to pinpoint what exactly do you love in a particular website and make note of it?

And then we can generate code for it, write css for it and deploy it on domain.


## Tech Stack

According to what I have identified yet, there would be 3 layers of this app.

1. User Facing Frontend : React TS, vite or TanStack start
2. Admin Panel : NextJs + TypeScript MonoRepo
3. User Backend : typeScript - will look into the framework
4. DB : MongoDb for Admin , PG or plain sql for llm 


## Basic Design
<img src="./inspiredIt Tech Design.png" width="100%" alt="tech design" />



## Plan of Action

1. Priority as of now for Frontend and LLM Engine. 
2. Will hold the Admin panel until prototype 1 is ready. 


