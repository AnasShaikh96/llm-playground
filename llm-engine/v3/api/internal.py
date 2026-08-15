from flask import Blueprint, Response
import json
from datetime import date


internal = Blueprint("internal", __name__)
internal.url_prefix = "/internal"


APPLICATION_JSON = 'application/json'

@internal.route("/")
def index():
  return Response(json.dumps("Internal API start"), mimetype=APPLICATION_JSON)


@internal.route("/chat")
def getChat():
  now = date.today()

  output = {
    "year":now.year,
    "month":now.year,
    "day":now.day
  }
  print(now)
  return Response(json.dumps(output),mimetype=APPLICATION_JSON)