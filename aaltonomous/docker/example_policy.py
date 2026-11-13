"""Example submission entrypoint: holds the arms still. Replace HoldPolicy with your policy."""
import logging

from aaltonomous.deploy.policy_server import HoldPolicy, serve

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve(HoldPolicy(), port=8000)
