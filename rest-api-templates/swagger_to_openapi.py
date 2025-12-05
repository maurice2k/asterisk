#!/usr/bin/env python3
"""
Convert Asterisk Swagger 1.x API specs to OpenAPI 3.1 JSON.

Usage:
    python swagger_to_openapi.py [--output-dir DIR]

Options:
    --output-dir DIR   Output directory (default: ./api-docs/openapi3)
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

# Try to import yaml, fall back to simple built-in serializer
try:
    import yaml
    YAML_AVAILABLE = 'pyyaml'
except ImportError:
    YAML_AVAILABLE = 'builtin'


def quote_yaml_key(key):
    """Quote YAML keys that need quoting (e.g., numeric strings)."""
    key_str = str(key)
    # Quote keys that are numeric or look like numbers
    if key_str.isdigit() or (key_str.startswith('-') and key_str[1:].isdigit()):
        return f"'{key_str}'"
    return key_str


def simple_yaml_dump(data, indent=0):
    """Simple YAML serializer for OpenAPI specs (no external dependencies)."""
    lines = []
    ind = '  ' * indent

    if isinstance(data, dict):
        if not data:
            return '{}'
        for key, value in data.items():
            quoted_key = quote_yaml_key(key)
            if isinstance(value, (dict, list)) and value:
                lines.append(f"{ind}{quoted_key}:")
                lines.append(simple_yaml_dump(value, indent + 1))
            elif isinstance(value, str):
                if '\n' in value:
                    # Multi-line string
                    lines.append(f"{ind}{quoted_key}: |")
                    for line in value.split('\n'):
                        lines.append(f"{ind}  {line}")
                elif any(c in value for c in ':{}[]#&*!|>\'"@`') or value == '':
                    # Quote strings with special characters
                    escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                    lines.append(f'{ind}{quoted_key}: "{escaped}"')
                else:
                    lines.append(f"{ind}{quoted_key}: {value}")
            elif isinstance(value, bool):
                lines.append(f"{ind}{quoted_key}: {str(value).lower()}")
            elif value is None:
                lines.append(f"{ind}{quoted_key}: null")
            else:
                lines.append(f"{ind}{quoted_key}: {value}")
    elif isinstance(data, list):
        if not data:
            return '[]'
        for item in data:
            if isinstance(item, dict):
                first = True
                for key, value in item.items():
                    prefix = f"{ind}- " if first else f"{ind}  "
                    first = False
                    quoted_key = quote_yaml_key(key)
                    if isinstance(value, (dict, list)) and value:
                        lines.append(f"{prefix}{quoted_key}:")
                        lines.append(simple_yaml_dump(value, indent + 2))
                    elif isinstance(value, str):
                        if '\n' in value:
                            lines.append(f"{prefix}{quoted_key}: |")
                            for line in value.split('\n'):
                                lines.append(f"{ind}    {line}")
                        elif any(c in value for c in ':{}[]#&*!|>\'"@`') or value == '':
                            escaped = value.replace('\\', '\\\\').replace('"', '\\"')
                            lines.append(f'{prefix}{quoted_key}: "{escaped}"')
                        else:
                            lines.append(f"{prefix}{quoted_key}: {value}")
                    elif isinstance(value, bool):
                        lines.append(f"{prefix}{quoted_key}: {str(value).lower()}")
                    elif value is None:
                        lines.append(f"{prefix}{quoted_key}: null")
                    else:
                        lines.append(f"{prefix}{quoted_key}: {value}")
            elif isinstance(item, str):
                if any(c in item for c in ':{}[]#&*!|>\'"@`'):
                    escaped = item.replace('\\', '\\\\').replace('"', '\\"')
                    lines.append(f'{ind}- "{escaped}"')
                else:
                    lines.append(f"{ind}- {item}")
            else:
                lines.append(f"{ind}- {item}")
    else:
        return str(data)

    return '\n'.join(lines)


SCRIPT_DIR = Path(__file__).parent.parent / 'rest-api'
SWAGGER_PRIMITIVES = ['void', 'string', 'boolean', 'number', 'int', 'long',
                      'double', 'float', 'Date', 'binary', 'object']


def load_json(filepath: Path) -> dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def convert_type(swagger_type: str, allow_multiple: bool = False) -> dict:
    """Convert Swagger 1.x type to OpenAPI 3 schema."""
    if not swagger_type:
        return {'type': 'string'}

    # Handle List[Type] syntax
    list_match = re.match(r'List\[(.+)\]', swagger_type)
    if list_match:
        inner_type = list_match.group(1)
        return {
            'type': 'array',
            'items': convert_type(inner_type)
        }

    # Handle primitives
    type_map = {
        'void': None,
        'string': {'type': 'string'},
        'boolean': {'type': 'boolean'},
        'number': {'type': 'number'},
        'int': {'type': 'integer'},
        'long': {'type': 'integer', 'format': 'int64'},
        'double': {'type': 'number', 'format': 'double'},
        'float': {'type': 'number', 'format': 'float'},
        'Date': {'type': 'string', 'format': 'date-time'},
        'binary': {'type': 'string', 'format': 'binary'},
        'object': {'type': 'object', 'additionalProperties': True},
        'containers': {'type': 'object', 'additionalProperties': True},
    }

    if swagger_type in type_map:
        result = type_map[swagger_type]
        if allow_multiple and result:
            return {'type': 'array', 'items': result}
        return result

    # Reference to a model
    ref = {'$ref': f'#/components/schemas/{swagger_type}'}
    if allow_multiple:
        return {'type': 'array', 'items': ref}
    return ref


def convert_parameter(param: dict) -> dict:
    """Convert Swagger 1.x parameter to OpenAPI 3 parameter or requestBody."""
    param_type = param.get('paramType', 'query')
    data_type = param.get('dataType', 'string')
    allow_multiple = param.get('allowMultiple', False)

    # Body parameters become requestBody (handled separately)
    if param_type == 'body':
        return None

    openapi_param = {
        'name': param.get('name'),
        'in': param_type,
        'description': param.get('description', ''),
    }

    # Path parameters MUST always have required: true in OpenAPI 3
    if param_type == 'path':
        openapi_param['required'] = True
    elif param.get('required', False):
        openapi_param['required'] = True

    # Build schema
    schema = convert_type(data_type, allow_multiple)
    if schema:
        # Handle allowableValues (enum)
        if 'allowableValues' in param:
            av = param['allowableValues']
            if av.get('valueType') == 'LIST' and 'values' in av:
                # For array types, put enum in items; for scalar types, put on schema
                if schema.get('type') == 'array' and 'items' in schema:
                    schema['items']['enum'] = av['values']
                else:
                    schema['enum'] = av['values']
            elif av.get('valueType') == 'RANGE':
                if 'min' in av:
                    schema['minimum'] = av['min']
                if 'max' in av:
                    schema['maximum'] = av['max']

        # Handle default value
        if 'defaultValue' in param:
            schema['default'] = param['defaultValue']

        # For array query parameters, add style and explode BEFORE schema
        # (to avoid YAML indentation issues with nested lists in schema)
        if param_type == 'query' and schema.get('type') == 'array':
            openapi_param['style'] = 'form'
            openapi_param['explode'] = False

        openapi_param['schema'] = schema

    return openapi_param


def convert_operation(operation: dict, path: str, resource_name: str = '') -> dict:
    """Convert Swagger 1.x operation to OpenAPI 3 operation."""
    method = operation.get('httpMethod', 'GET').lower()

    # Create unique operationId by prefixing with resource name
    nickname = operation.get('nickname', '')
    operation_id = f"{resource_name}_{nickname}" if resource_name and nickname else nickname

    openapi_op = {
        'summary': operation.get('summary', ''),
        'operationId': operation_id,
    }

    if operation.get('notes'):
        openapi_op['description'] = operation['notes']

    # Convert 'since' to x-version-added
    if operation.get('since'):
        since = operation['since']
        # since is an array, take the first version
        openapi_op['x-version-added'] = since[0] if isinstance(since, list) else since

    # Convert WebSocket fields to x- extensions
    if operation.get('upgrade') == 'websocket':
        openapi_op['x-upgrade'] = 'websocket'
        if operation.get('websocketProtocol'):
            openapi_op['x-websocket-protocol'] = operation['websocketProtocol']

    # Note: x-requires-modules is added at path level, not operation level

    # Convert parameters - track names for body param insertion point
    params = []
    request_body = None
    prev_param_name = None  # Track previous parameter name for body insertion

    for param in operation.get('parameters', []):
        if param.get('paramType') == 'body':
            # Handle body parameter as requestBody
            data_type = param.get('dataType', 'object')
            param_name = param.get('name', '')
            schema = convert_type(data_type)
            if schema:
                # Use VariableBag schema for 'variables' or 'containers' type
                if param_name == 'variables' or data_type == 'containers':
                    body_schema = {'$ref': '#/components/schemas/VariableBag'}
                else:
                    body_schema = schema
                request_body = {
                    'content': {
                        'application/json': {
                            'schema': body_schema
                        }
                    }
                }
                # Preserve body parameter name for code generation
                if param_name:
                    request_body['x-body-name'] = param_name
                # Track insertion point: name of previous param, or null if first
                request_body['x-param-insert-after'] = prev_param_name
                # Include description from body parameter
                if param.get('description'):
                    request_body['description'] = param['description']
                if param.get('required'):
                    request_body['required'] = True
        else:
            converted = convert_parameter(param)
            if converted:
                params.append(converted)
                prev_param_name = param.get('name')  # Update previous param name

    if params:
        openapi_op['parameters'] = params

    if request_body:
        openapi_op['requestBody'] = request_body

    # Convert responses
    response_class = operation.get('responseClass', 'void')
    responses = {}

    # Success response (matching actual Asterisk response_text values)
    if response_class == 'void':
        responses['204'] = {'description': 'No Content'}
    else:
        schema = convert_type(response_class)
        if schema:
            responses['200'] = {
                'description': 'OK',
                'content': {
                    'application/json': {'schema': schema}
                }
            }
        else:
            responses['200'] = {'description': 'OK'}

    # Error responses
    for error in operation.get('errorResponses', []):
        code = str(error.get('code', 500))
        responses[code] = {'description': error.get('reason', 'Error')}

    openapi_op['responses'] = responses

    return method, openapi_op


def convert_model(model_name: str, model: dict, all_models: dict = None) -> dict:
    """Convert Swagger 1.x model to OpenAPI 3 schema.

    Args:
        model_name: Name of this model
        model: The model definition from Swagger
        all_models: Dict of all models (needed to find parent for inheritance)
    """
    all_models = all_models or {}

    # Find if this model has a parent (is listed in another model's subTypes)
    parent_name = None
    for other_name, other_model in all_models.items():
        if model_name in other_model.get('subTypes', []):
            parent_name = other_name
            break

    # Build the schema for this model's own properties
    own_schema = {
        'type': 'object',
    }

    if model.get('description'):
        own_schema['description'] = model['description']

    properties = {}
    required = []

    for prop_name, prop in model.get('properties', {}).items():
        prop_schema = {}

        # Get type
        prop_type = prop.get('type', 'string')
        prop_schema = convert_type(prop_type) or {'type': 'string'}

        # Copy description
        if prop.get('description'):
            prop_schema['description'] = prop['description']

        # Handle enum
        if 'allowableValues' in prop:
            av = prop['allowableValues']
            if av.get('valueType') == 'LIST' and 'values' in av:
                prop_schema['enum'] = av['values']

        properties[prop_name] = prop_schema

        # Track required fields
        if prop.get('required', False):
            required.append(prop_name)

    if properties:
        own_schema['properties'] = properties

    if required:
        own_schema['required'] = required

    # Handle discriminator for subtypes
    if model.get('subTypes'):
        # Find the discriminator property - either on this model or inherited from parent
        discriminator_prop = model.get('discriminator')
        if not discriminator_prop and parent_name:
            # Look up the chain for discriminator
            current = all_models.get(parent_name)
            while current:
                if current.get('discriminator'):
                    discriminator_prop = current['discriminator']
                    break
                # Find parent of current
                next_parent = None
                for other_name, other_model in all_models.items():
                    if current.get('id', '') in other_model.get('subTypes', []):
                        next_parent = other_name
                        break
                current = all_models.get(next_parent) if next_parent else None

        if discriminator_prop:
            discriminator = {
                'propertyName': discriminator_prop
            }
            # Build mapping from subtype name to schema ref
            mapping = {}
            for subtype in model.get('subTypes', []):
                mapping[subtype] = f'#/components/schemas/{subtype}'
            if mapping:
                discriminator['mapping'] = mapping
            own_schema['discriminator'] = discriminator

            # For models with their own discriminator, add enum to the type property
            if model.get('discriminator') == discriminator_prop:
                if discriminator_prop in own_schema.get('properties', {}):
                    own_schema['properties'][discriminator_prop]['enum'] = list(model['subTypes'])

    # If this model has a parent, use allOf for inheritance
    if parent_name:
        # Check if any ancestor has a discriminator - if so, add const for type
        discriminator_prop = None
        current = all_models.get(parent_name)
        while current:
            if current.get('discriminator'):
                discriminator_prop = current['discriminator']
                break
            # Find parent of current
            next_parent = None
            for other_name, other_model in all_models.items():
                if current.get('id', '') in other_model.get('subTypes', []):
                    next_parent = other_name
                    break
            current = all_models.get(next_parent) if next_parent else None

        # Add discriminator property value
        has_own_subtypes = model.get('subTypes')
        own_discriminator = model.get('discriminator')

        if discriminator_prop:
            if 'properties' not in own_schema:
                own_schema['properties'] = {}

            if has_own_subtypes:
                # Model has subtypes: add enum with all possible subtype values
                own_schema['properties'][discriminator_prop] = {
                    'type': 'string',
                    'enum': list(has_own_subtypes)
                }
            elif not own_discriminator:
                # Leaf model: use const for single value
                own_schema['properties'][discriminator_prop] = {
                    'type': 'string',
                    'const': model_name
                }

        schema = {
            'allOf': [
                {'$ref': f'#/components/schemas/{parent_name}'},
                own_schema
            ]
        }
        # Move description to top level if present
        if own_schema.get('description'):
            schema['description'] = own_schema.pop('description')
        return schema

    return own_schema


def convert_api_declaration(api_doc: dict, tag_name: str) -> tuple:
    """Convert Swagger 1.x API declaration to OpenAPI 3 paths and schemas."""
    paths = {}
    schemas = {}

    # Get requiresModules from API declaration level
    requires_modules = api_doc.get('requiresModules', [])

    # Convert APIs to paths
    for api in api_doc.get('apis', []):
        path = api.get('path', '')
        path_item = {}

        # Add path-level description if present
        if api.get('description'):
            path_item['description'] = api['description']

        # Add x-requires-modules at path level (from API declaration)
        if requires_modules:
            path_item['x-requires-modules'] = requires_modules

        for operation in api.get('operations', []):
            method, openapi_op = convert_operation(
                operation, path, tag_name
            )
            openapi_op['tags'] = [tag_name]
            path_item[method] = openapi_op

        if path_item:
            paths[path] = path_item

    # Convert models to schemas (pass all_models for inheritance resolution)
    # Add x-source-api to track which API file defined each model
    all_models = api_doc.get('models', {})
    for model_name, model in all_models.items():
        schema = convert_model(model_name, model, all_models)
        schema['x-source-api'] = tag_name
        schemas[model_name] = schema

    return paths, schemas


def generate_openapi(resources_file: Path, api_docs_dir: Path) -> dict:
    """Generate complete OpenAPI 3 spec from Swagger 1.x files."""
    resources = load_json(resources_file)

    # Default copyright/author for the whole spec
    default_copyright = 'Copyright (C) 2012 - 2013, Digium, Inc.'
    default_author = 'David M. Lee, II <dlee@digium.com>'

    openapi = {
        'openapi': '3.1.2',
        'info': {
            'title': 'Asterisk REST Interface (ARI)',
            'description': 'REST interface for Asterisk.',
            'version': resources.get('apiVersion', '1.0.0'),
            'license': {
                'name': 'GPL-2.0',
                'url': 'https://www.gnu.org/licenses/old-licenses/gpl-2.0.html'
            },
            'x-copyright': default_copyright,
            'x-author': default_author
        },
        'servers': [
            {
                'url': resources.get('basePath', 'http://localhost:8088/ari'),
                'description': 'Asterisk ARI server'
            }
        ],
        'tags': [],
        'paths': {},
        'components': {
            'securitySchemes': {
                'basicAuth': {
                    'type': 'http',
                    'scheme': 'basic'
                }
            },
            'schemas': {}
        },
        'security': [{'basicAuth': []}]
    }

    # Process each API resource
    for api_ref in resources.get('apis', []):
        # Extract filename from path like "/api-docs/asterisk.{format}"
        api_path = api_ref.get('path', '')
        match = re.search(r'/api-docs/(\w+)\.', api_path)
        if not match:
            continue

        api_name = match.group(1)
        api_file = api_docs_dir / f'{api_name}.json'

        if not api_file.exists():
            print(f"Warning: {api_file} not found, skipping", file=sys.stderr)
            continue

        # Load and convert API declaration
        api_doc = load_json(api_file)

        # Add tag (only include copyright/author if different from global)
        tag = {
            'name': api_name,
            'description': api_ref.get('description', '')
        }
        api_copyright = api_doc.get('_copyright')
        api_author = api_doc.get('_author')
        if api_copyright and api_copyright != default_copyright:
            tag['x-copyright'] = api_copyright
        if api_author and api_author != default_author:
            tag['x-author'] = api_author
        openapi['tags'].append(tag)
        paths, schemas = convert_api_declaration(api_doc, api_name)

        # Merge paths and schemas
        openapi['paths'].update(paths)
        openapi['components']['schemas'].update(schemas)

    # Add VariableBag schema (used for request bodies with variables)
    openapi['components']['schemas']['VariableBag'] = {
        'type': 'object',
        'description': 'Container for key/value variable pairs',
        'properties': {
            'variables': {
                'type': 'object',
                'description': 'Key/value pairs of variables',
                'additionalProperties': {'type': 'string'}
            }
        }
    }

    # Sort schemas alphabetically for consistent ordering
    openapi['components']['schemas'] = dict(
        sorted(openapi['components']['schemas'].items())
    )

    return openapi


if YAML_AVAILABLE == 'pyyaml':
    class NoAliasDumper(yaml.SafeDumper):
        """Custom YAML dumper that disables anchors/aliases."""
        def ignore_aliases(self, data):
            return True
else:
    NoAliasDumper = None


def write_output(data: dict, filepath: Path, format: str):
    """Write output file in specified format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        if format == 'yaml':
            if YAML_AVAILABLE == 'pyyaml':
                yaml.dump(data, f, Dumper=NoAliasDumper,
                         default_flow_style=False, sort_keys=False,
                         allow_unicode=True, width=120, indent=2)
            else:
                f.write(simple_yaml_dump(data))
                f.write('\n')
        else:
            json.dump(data, f, indent=2)

    print(f"Written: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Asterisk Swagger 1.x specs to OpenAPI 3.1'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path(__file__).parent.parent / 'rest-api' / 'api-docs' / 'openapi3',
        help='Output directory (default: ../rest-api/api-docs/openapi3)'
    )
    parser.add_argument(
        '--input-dir', '-i',
        type=Path,
        default=Path(__file__).parent.parent / 'rest-api',
        help='Input directory containing resources.json and api-docs/'
    )

    args = parser.parse_args()

    resources_file = args.input_dir / 'resources.json'
    api_docs_dir = args.input_dir / 'api-docs'

    if not resources_file.exists():
        print(f"Error: {resources_file} not found", file=sys.stderr)
        sys.exit(1)

    if not api_docs_dir.exists():
        print(f"Error: {api_docs_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate OpenAPI spec
    openapi = generate_openapi(resources_file, api_docs_dir)

    # Write JSON only
    json_file = args.output_dir / 'openapi3.json'

    write_output(openapi, json_file, 'json')

    print(f"\nDone! OpenAPI 3.1 spec generated in {args.output_dir}")


if __name__ == '__main__':
    main()

